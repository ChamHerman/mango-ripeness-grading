import cv2
import numpy as np
import time
import os
import joblib

MODEL_PATH = "output/geometry_based/geometry_based_model.joblib"
_CACHED_MODEL = None

def load_geometry_model():
    """Load the trained geometry/edge hybrid model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
        
    resolved_path = MODEL_PATH
    if not os.path.exists(resolved_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", MODEL_PATH)
        if os.path.exists(alt_path):
            resolved_path = alt_path
            
    if os.path.exists(resolved_path):
        _CACHED_MODEL = joblib.load(resolved_path)
        return _CACHED_MODEL
    
    return None

def get_preprocessing_masks(img_bgr):
    """Generates the background mask and interior erosion mask."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    se_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    int_mask = cv2.erode(mask, se_erode)
    return gray, mask, int_mask

def extract_geometric_features(img_bgr):
    """Extracts Scharr edge density and structural geometric features."""
    gray, mask, int_mask = get_preprocessing_masks(img_bgr)
    area = np.sum(mask > 0)
    
    feats = {
        'circularity': 0.0,
        'aspect_ratio': 0.0,
        'convexity_defects': 0.0,
        'avg_sobel_gradient': 0.0,
        'scharr_density': 0.0
    }
    
    if area == 0:
        return feats, gray, mask, int_mask, None, None
        
    # Scharr Edge Density
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_mag = np.uint8(np.absolute(scharrx) + np.absolute(scharry))
    _, scharr_edges = cv2.threshold(scharr_mag, 50, 255, cv2.THRESH_BINARY)
    scharr_edges = cv2.bitwise_and(scharr_edges, scharr_edges, mask=int_mask)
    feats['scharr_density'] = float(np.sum(scharr_edges > 0) / (np.sum(int_mask > 0) + 1e-6))
    
    # Geometric Shape Analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        largest_contour = c
        perimeter = cv2.arcLength(c, True)
        feats['circularity'] = float(4 * np.pi * (area / (perimeter * perimeter + 1e-6)))
        
        x, y, w, h = cv2.boundingRect(c)
        feats['aspect_ratio'] = float(float(w) / h if h > 0 else 0.0)
        
        hull = cv2.convexHull(c, returnPoints=False)
        defects_depth_sum = 0
        if hull is not None and len(hull) > 3 and len(c) > 3:
            try:
                defects = cv2.convexityDefects(c, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        defects_depth_sum += d / 256.0
            except Exception:
                pass
        feats['convexity_defects'] = float(defects_depth_sum / (area + 1e-6))
        
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    feats['avg_sobel_gradient'] = float(np.mean(sobel_mag[int_mask > 0])) if np.sum(int_mask > 0) > 0 else 0.0
    
    return feats, gray, mask, int_mask, scharr_edges, largest_contour

def analyze_ripeness_by_geometry(image: np.ndarray):
    """
    Yeow Wei Kang's Edge & Shape Deformity Detection module.
    Runs inference using the loaded geometry_based_model.joblib SVM.
    """
    t_start = time.time()
    
    model_pkg = load_geometry_model()
    
    if not model_pkg:
        # Fallback if model missing
        return "unripe", 0.0, image, {}, {}
        
    scaler = model_pkg['scaler']
    model = model_pkg['model']
    features_list = model_pkg['features']
    classes = model_pkg['classes']
    
    feats, gray, mask, int_mask, scharr_edges, largest_contour = extract_geometric_features(image)
    
    # Predict
    input_vector = []
    for f in features_list:
        input_vector.append(feats.get(f, 0.0))
        
    input_scaled = scaler.transform([input_vector])
    
    prediction_idx = model.predict(input_scaled)[0]
    prediction = classes[prediction_idx]
    
    # Calculate confidence from probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probs) * 100.0)
    else:
        confidence = 100.0
        
    # Visualisation
    visualized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(visualized, (x, y), (x + w, y + h), (0, 191, 255), 2)
        cv2.drawContours(visualized, [largest_contour], -1, (255, 215, 0), 2)
        
    if scharr_edges is not None:
        visualized[scharr_edges > 0] = [255, 69, 0]
        
    latency_ms = (time.time() - t_start) * 1000.0
    
    metrics = {
        'scharr_density': round(feats.get('scharr_density', 0.0), 4),
        'circularity': round(feats.get('circularity', 0.0), 4),
        'aspect_ratio': round(feats.get('aspect_ratio', 0.0), 4),
        'convexity_defects': round(feats.get('convexity_defects', 0.0), 6),
        'avg_sobel_gradient': round(feats.get('avg_sobel_gradient', 0.0), 4),
        'latency_ms': round(latency_ms, 2)
    }
    
    step_images = {
        '1. Preprocessing Mask': mask,
        '2. Interior Erosion Mask': int_mask,
        '3. Scharr Edges': scharr_edges if scharr_edges is not None else np.zeros_like(gray),
        '4. Contour Bounding': visualized
    }
    
    return prediction, float(confidence), visualized, metrics, step_images
