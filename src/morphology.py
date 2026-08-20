import os
import time
import cv2
import numpy as np
import pandas as pd
import joblib

def get_fruit_mask(img_bgr, thresh=20):
    """Extract and clean foreground mango mask from black background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thresh).astype(np.uint8) * 255
    se_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_clean)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_clean)
    return mask

def get_interior_mask(mask, k=13):
    """Erode mango mask to eliminate boundary transition ring (ADR-0001)."""
    se_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, se_erode)

def run_enhanced_morphology_pipeline(gray, interior):
    """
    Enhanced Multi-Scale Morphological Hybrid Pipeline.
    Combines multi-scale Beucher gradient, dual-scale black-hat, and morphological noise filtering.
    """
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    se15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # 1. Beucher Gradient at scale 3
    grad = cv2.subtract(cv2.dilate(gray, se3), cv2.erode(gray, se3))
    grad = cv2.bitwise_and(grad, grad, mask=interior)
    
    # 2. Multi-scale Black-Hat at scale 7 and 15
    bh7 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se7)
    bh15 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se15)
    bh = cv2.bitwise_or(bh7, bh15)
    bh = cv2.bitwise_and(bh, bh, mask=interior)
    
    # 3. Dynamic Thresholding
    vals_grad = grad[interior > 0]
    thresh_grad = max(np.percentile(vals_grad, 92) if len(vals_grad) > 0 else 25, 20)
    _, bw_grad = cv2.threshold(grad, thresh_grad, 255, cv2.THRESH_BINARY)
    
    _, bw_bh = cv2.threshold(bh, 22, 255, cv2.THRESH_BINARY)
    
    # 4. Morphological Fusion & Closing
    bw = cv2.bitwise_or(bw_grad, bw_bh)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se3)
    bw = cv2.bitwise_and(bw, bw, mask=interior)
    
    # 5. Connected Component Area Filtering (< 10 px)
    nb, out, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    sizes = stats[1:, -1]
    bw_clean = np.zeros_like(bw)
    for i in range(0, nb - 1):
        if sizes[i] >= 10:
            bw_clean[out == i + 1] = 255
            
    bw_clean = cv2.bitwise_and(bw_clean, bw_clean, mask=interior)
    return bw_clean, {
        'grad': grad, 'bh': bh, 'bw_grad': bw_grad, 'bw_bh': bw_bh, 'bw_clean': bw_clean
    }

def extract_morphological_features(gray, bw_blemish, interior_mask, interior_area):
    """Extract standardised 10-dimensional morphological feature vector."""
    feats = {
        'n_blemishes': 0,
        'blemish_area': 0,
        'area_ratio': 0.0,
        'mean_darkness': 0.0,
        'max_darkness': 0.0,
        'mean_brightness': 0.0,
        'mean_aspect_ratio': 0.0,
        'mean_solidity': 0.0,
        'skeleton_length': 0,
        'blemish_dispersion': 0.0
    }
    
    if interior_area == 0:
        return feats

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_blemish, connectivity=8)
    if num_labels <= 1:
        return feats

    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    
    feats['n_blemishes'] = int(num_labels - 1)
    feats['blemish_area'] = int(np.sum(areas))
    feats['area_ratio'] = float((feats['blemish_area'] / interior_area) * 100.0)

    blemish_pixels = gray[bw_blemish > 0]
    if len(blemish_pixels) > 0:
        feats['mean_darkness'] = float(np.mean(255.0 - blemish_pixels))
        feats['max_darkness'] = float(np.max(255.0 - blemish_pixels))
        feats['mean_brightness'] = float(np.mean(blemish_pixels))

    aspect_ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-5)
    feats['mean_aspect_ratio'] = float(np.mean(aspect_ratios))
    
    bbox_areas = widths * heights
    solidity_est = areas / (bbox_areas + 1e-5)
    feats['mean_solidity'] = float(np.mean(np.clip(solidity_est, 0.0, 1.0)))

    if len(centroids) > 1:
        c_pts = centroids[1:]
        fruit_center = np.mean(c_pts, axis=0)
        disp = np.mean(np.linalg.norm(c_pts - fruit_center, axis=1))
        feats['blemish_dispersion'] = float(disp)

    skel = np.zeros(bw_blemish.shape, np.uint8)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = bw_blemish.copy()
    for _ in range(4):
        eroded = cv2.erode(temp_img, elem)
        temp = cv2.dilate(eroded, elem)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break
    feats['skeleton_length'] = int(cv2.countNonZero(skel))

    return feats

def analyze_ripeness_by_morphology(image_bgr: np.ndarray, model_path: str = "output/morphology_based/morphology_model.joblib"):
    """
    Herman's pure Mathematical Morphology Blemish Analysis inference function.
    
    Returns:
        prediction (str): 'unripe', 'fully_ripe', or 'overripe'
        confidence (float): Classification confidence percentage (0-100%)
        visualized_img (np.ndarray): Image with red blemish overlay in RGB
        metrics (dict): Morphological metrics dictionary (blemish area %, count, etc.)
    """
    t_start = time.time()
    
    if not os.path.exists(model_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Morphology model package not found at: {model_path}")
            
    pkg = joblib.load(model_path)
    model = pkg['model']
    feature_cols = pkg['feature_cols']
    
    # 1. Masking & Preprocessing
    mask = get_fruit_mask(image_bgr)
    interior = get_interior_mask(mask)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    interior_area = int(np.sum(interior > 0))
    
    # 2. Enhanced Morphology Pipeline
    bw_clean, vis = run_enhanced_morphology_pipeline(gray, interior)
    feats = extract_morphological_features(gray, bw_clean, interior, interior_area)
    
    # 3. Model Prediction
    feat_df = pd.DataFrame([feats])[feature_cols]
    pred_cls = model.predict(feat_df)[0]
    pred_prob = model.predict_proba(feat_df)[0]
    conf = float(np.max(pred_prob) * 100.0)
    latency_ms = (time.time() - t_start) * 1000.0
    
    # 4. Visualization Overlay
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()
    overlay[bw_clean > 0] = [255, 0, 0] # Red blemish mask
    blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
    
    metrics = {
        'blemish_area_ratio': feats['area_ratio'],
        'n_blemishes': feats['n_blemishes'],
        'mean_darkness': feats['mean_darkness'],
        'skeleton_length': feats['skeleton_length'],
        'latency_ms': latency_ms,
        'features': feats
    }
    
    step_images = {
        '1. Eroded Fruit Mask': interior,
        '2. Beucher Gradient (SE 3x3)': vis['grad'],
        '3. Multi-Scale Black-Hat': vis['bh'],
        '4. Fused Blemish Mask': bw_clean,
        '5. Blemish Overlay': blended
    }
    
    return pred_cls, conf, blended, metrics, step_images
