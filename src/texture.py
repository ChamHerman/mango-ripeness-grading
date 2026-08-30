import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Global cache for trained texture model package
_CACHED_MODEL_PKG = None
DEFAULT_MODEL_PATH = "output/texture_based/texture_model.joblib"

# Hyperparameters matching Mode A notebook
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
LBP_P = 8
LBP_R = 1
LBP_METHOD = 'uniform'


def load_texture_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and cache the trained texture model package."""
    global _CACHED_MODEL_PKG
    if _CACHED_MODEL_PKG is not None:
        return _CACHED_MODEL_PKG

    resolved_path = model_path
    if not os.path.exists(resolved_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", model_path)
        if os.path.exists(alt_path):
            resolved_path = alt_path
        else:
            raise FileNotFoundError(f"Texture model package not found at: {model_path}")

    _CACHED_MODEL_PKG = joblib.load(resolved_path)
    return _CACHED_MODEL_PKG


def get_fruit_mask(img_bgr: np.ndarray, thresh: int = 20) -> np.ndarray:
    """Extract foreground mango binary mask from black background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thresh).astype(np.uint8) * 255
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    return mask


def extract_glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Compute rotation-invariant GLCM texture features on fruit ROI."""
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    glcm = graycomatrix(
        masked_gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=False
    )
    # Zero-out background pair counts at level 0
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    total = glcm.sum()
    if total > 0:
        glcm = glcm.astype(np.float64) / total

    return {
        'glcm_contrast': float(np.mean(graycoprops(glcm, 'contrast'))),
        'glcm_correlation': float(np.mean(graycoprops(glcm, 'correlation'))),
        'glcm_energy': float(np.mean(graycoprops(glcm, 'energy'))),
        'glcm_homogeneity': float(np.mean(graycoprops(glcm, 'homogeneity')))
    }


def extract_lbp_features(gray: np.ndarray, mask: np.ndarray, P: int = LBP_P, R: int = LBP_R) -> tuple:
    """Compute Local Binary Pattern (LBP) statistics on fruit region."""
    lbp = local_binary_pattern(gray, P=P, R=R, method=LBP_METHOD)
    fruit_lbp = lbp[mask > 0]

    if len(fruit_lbp) == 0:
        return {'lbp_mean': 0.0, 'lbp_variance': 0.0, 'lbp_entropy': 0.0}, lbp

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(fruit_lbp, bins=n_bins, range=(0, n_bins), density=True)
    hist = hist[hist > 0]
    entropy = -float(np.sum(hist * np.log2(hist)))

    feats = {
        'lbp_mean': float(np.mean(fruit_lbp)),
        'lbp_variance': float(np.var(fruit_lbp)),
        'lbp_entropy': entropy
    }
    return feats, lbp


def extract_spatial_roughness(gray: np.ndarray, mask: np.ndarray) -> tuple:
    """Compute Sobel spatial gradient surface roughness."""
    denoised = cv2.medianBlur(gray, 3)
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    sobel = cv2.bitwise_and(sobel, sobel, mask=mask)
    
    fruit_grad = sobel[mask > 0]
    roughness = float(np.mean(fruit_grad)) if len(fruit_grad) > 0 else 0.0
    return roughness, sobel


def analyze_ripeness_by_texture(image: np.ndarray, model_path: str = DEFAULT_MODEL_PATH):
    """
    Wong Kai Bin's Enhanced Multi-Descriptor Texture Analysis production inference module.
    Combines Rotation-Invariant GLCM + Uniform LBP + Sobel Surface Roughness.
    
    Parameters:
        image (np.ndarray): Input BGR mango image.
        model_path (str): Path to serialized model package (texture_model.joblib).
        
    Returns:
        prediction (str): Ripeness stage ('unripe', 'fully_ripe', or 'overripe').
        confidence (float): Classification confidence percentage (0-100%).
        visualized_img (np.ndarray): RGB texture gradient & surface overlay image.
        metrics (dict): Extracted GLCM & LBP metrics dictionary.
        step_images (dict): Dictionary of intermediate pipeline stage images.
    """
    t_start = time.time()

    # 1. Load trained ML package
    pkg = load_texture_model(model_path)
    model = pkg['model']
    scaler = pkg['scaler']
    feature_cols = pkg['feature_cols']
    label_map = pkg['label_map']
    inv_label_map = {v: k for k, v in label_map.items()}

    # 2. Fruit Masking & Grayscale Intensity
    mask = get_fruit_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Extract 8-D Enhanced Texture Features
    glcm_feats = extract_glcm_features(gray, mask)
    lbp_feats, lbp_map = extract_lbp_features(gray, mask)
    roughness, sobel_map = extract_spatial_roughness(gray, mask)
    
    combined_feats = {
        **glcm_feats,
        **lbp_feats,
        'surface_roughness': roughness
    }

    # 4. Standard Scaling & ML Inference
    feat_df = pd.DataFrame([combined_feats])[feature_cols]
    scaled_feats = scaler.transform(feat_df.values)

    pred_label = int(model.predict(scaled_feats)[0])
    prediction = inv_label_map.get(pred_label, "unknown")

    # Compute classification confidence
    confidence = 90.0
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(scaled_feats)[0]
        confidence = float(np.max(probs)) * 100.0
    elif hasattr(model, 'decision_function'):
        df_val = model.decision_function(scaled_feats)
        exp_vals = np.exp(df_val - np.max(df_val))
        softmax_probs = exp_vals / np.sum(exp_vals)
        confidence = float(np.max(softmax_probs)) * 100.0

    latency_ms = (time.time() - t_start) * 1000.0

    # 5. Visualizations & Step Images
    heatmap = cv2.applyColorMap(sobel_map, cv2.COLORMAP_JET)
    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb, 0.65, heatmap_rgb, 0.35, 0)

    norm_lbp = np.uint8(np.clip(lbp_map / (LBP_P + 2) * 255.0, 0, 255))
    norm_lbp = cv2.bitwise_and(norm_lbp, norm_lbp, mask=mask)
    lbp_colored = cv2.applyColorMap(norm_lbp, cv2.COLORMAP_MAGMA)
    lbp_colored = cv2.bitwise_and(lbp_colored, lbp_colored, mask=mask)
    lbp_colored_rgb = cv2.cvtColor(lbp_colored, cv2.COLOR_BGR2RGB)

    metrics = {
        'glcm_contrast': round(combined_feats['glcm_contrast'], 2),
        'glcm_homogeneity': round(combined_feats['glcm_homogeneity'], 4),
        'glcm_energy': round(combined_feats['glcm_energy'], 4),
        'glcm_correlation': round(combined_feats['glcm_correlation'], 4),
        'lbp_mean': round(combined_feats['lbp_mean'], 2),
        'lbp_variance': round(combined_feats['lbp_variance'], 2),
        'lbp_entropy': round(combined_feats['lbp_entropy'], 2),
        'surface_roughness': round(roughness, 2),
        'classifier': pkg.get('classifier', 'KNN (k=5)'),
        'test_accuracy': pkg.get('test_accuracy', 92.36),
        'latency_ms': round(latency_ms, 2),
        'features': combined_feats
    }

    step_images = {
        '1. Grayscale Intensity Map': gray,
        '2. Fruit Segmentation Mask': mask,
        '3. Local Binary Pattern (LBP) Map': lbp_colored_rgb,
        '4. GLCM Spatial Texture Heatmap': heatmap_rgb,
        '5. Texture Gradient Overlay': blended
    }

    return prediction, float(confidence), blended, metrics, step_images
