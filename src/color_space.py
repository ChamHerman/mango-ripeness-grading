import os
import cv2
import numpy as np
import pandas as pd
import time
import joblib

MODEL_PATH = "output/color_based/models/HSV_model.pkl"
SCALER_PATH = "output/color_based/models/HSV_scaler.pkl"

def extract_channel_statistics(channel, mask, prefix):
    pixels = channel[mask > 0]
    if len(pixels) == 0:
        return {
            f"{prefix}_Mean": 0.0, f"{prefix}_Std": 0.0, f"{prefix}_Median": 0.0
        }
    return {
        f"{prefix}_Mean": float(np.mean(pixels)),
        f"{prefix}_Std": float(np.std(pixels)),
        f"{prefix}_Median": float(np.median(pixels))
    }

def analyze_ripeness_by_color(image: np.ndarray):
    """
    Lum Siew Feng's completed Color-Space Analysis module.
    Extracts 9 HSV statistical channel features and evaluates with trained SVC model.
    
    Returns:
        prediction (str): 'unripe', 'fully_ripe', or 'overripe'
        confidence (float): Confidence score (0-100%)
        visualized_img (np.ndarray): Color-segmented visualization image (RGB)
        metrics (dict): Extracted color space metrics
        step_images (dict): Dictionary of intermediate pipeline stage images
    """
    t_start = time.time()
    
    # 1. Masking
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 20).astype(np.uint8) * 255
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    
    # 2. HSV Conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv)
    channel_names = ["H", "S", "V"]
    
    features = {}
    for ch, name in zip(channels, channel_names):
        prefix = f"HSV_{name}"
        features.update(extract_channel_statistics(ch, mask, prefix))
        
    feature_order = ['HSV_H_Mean', 'HSV_H_Std', 'HSV_H_Median', 'HSV_S_Mean', 'HSV_S_Std', 'HSV_S_Median', 'HSV_V_Mean', 'HSV_V_Std', 'HSV_V_Median']
    feat_df = pd.DataFrame([features])[feature_order]
    
    # 3. Model Inference using Siew Feng's saved model
    model_file = MODEL_PATH
    scaler_file = SCALER_PATH
    if not os.path.exists(model_file):
        alt_m = os.path.join(os.path.dirname(__file__), "..", MODEL_PATH)
        alt_s = os.path.join(os.path.dirname(__file__), "..", SCALER_PATH)
        if os.path.exists(alt_m):
            model_file = alt_m
            scaler_file = alt_s
            
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        X_scaled = scaler.transform(feat_df)
        pred_cls = model.predict(X_scaled)[0]
        
        # Calculate decision function confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            conf = float(np.max(probs) * 100.0)
        else:
            d = model.decision_function(X_scaled)[0]
            exp_d = np.exp(d - np.max(d))
            probs = exp_d / np.sum(exp_d)
            conf = float(np.max(probs) * 100.0)
    else:
        mean_h = features.get('HSV_H_Mean', 0)
        if mean_h > 45:
            pred_cls = 'unripe'
            conf = 85.0
        elif mean_h > 20:
            pred_cls = 'fully_ripe'
            conf = 85.0
        else:
            pred_cls = 'overripe'
            conf = 80.0
            
    latency_ms = (time.time() - t_start) * 1000.0
    
    # 4. Color Segmentation Masks for Intermediate Pipeline
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.bitwise_and(mask_green, mask_green, mask=mask)
    
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([29, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow = cv2.bitwise_and(mask_yellow, mask_yellow, mask=mask)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualized = img_rgb.copy()
    visualized[mask_green > 0] = [34, 139, 34]    # Green
    visualized[mask_yellow > 0] = [255, 215, 0]  # Yellow
    blended = cv2.addWeighted(img_rgb, 0.6, visualized, 0.4, 0)
    
    hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    fruit_area = np.sum(mask > 0)
    green_pct = float(np.sum(mask_green > 0) / fruit_area * 100.0) if fruit_area > 0 else 0.0
    yellow_pct = float(np.sum(mask_yellow > 0) / fruit_area * 100.0) if fruit_area > 0 else 0.0
    
    metrics = {
        'mean_hue': round(features.get('HSV_H_Mean', 0), 2),
        'mean_saturation': round(features.get('HSV_S_Mean', 0), 2),
        'mean_value': round(features.get('HSV_V_Mean', 0), 2),
        'green_coverage_pct': round(green_pct, 2),
        'yellow_coverage_pct': round(yellow_pct, 2),
        'latency_ms': round(latency_ms, 2)
    }
    
    step_images = {
        '1. Foreground Fruit Mask': mask,
        '2. HSV Color Space Representation': hsv_rgb,
        '3. Green Peel Mask (Unripe)': mask_green,
        '4. Yellow Peel Mask (Ripe)': mask_yellow,
        '5. Segmented Color Overlay': blended
    }
    
    return pred_cls, float(conf), blended, metrics, step_images
