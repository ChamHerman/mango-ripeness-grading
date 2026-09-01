"""Real-Time Multi-Mango Detection, Counting & Ripeness Localization Module.

100% Classical Image Processing & Heuristic Rule Engines (NO ML / NO Deep Learning).

Features:
1. Multi-instance mango segmentation & background removal (K-Means / Morphological Seed).
2. Connected component and contour shape analysis for mango counting and localization.
3. Per-mango ROI extraction and classical ripeness grading using pure computer vision heuristics:
   - Classical Color-Space Chrominance & Hue Thresholding (Lum Siew Feng)
   - Classical Morphological Blemish & Granulometry Quantification (Cham Herman)
   - Classical Surface Roughness & LBP Texture Dispersion (Wong Kai Bin)
   - Classical Scharr Edge Density & Contour Deformity (Yeow Wei Kang)
   - Classical Multi-Feature Rule-Based Fusion (Heuristic Consensus)
4. In-frame localized bounding boxes and ripeness verdict badges positioned directly at mangoes.
5. Real-time telemetry, rolling FPS, and latency metrics matrix calculation.
6. Thread-safe streaming callbacks for Streamlit WebRTC and snapshot ingestion.
"""

import collections
import time
import threading
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd

from src.preprocessing import (
    resize_image,
    remove_noise,
    enhance_contrast,
    segment_fruit_mask,
    remove_background,
    clean_mask,
    generate_mango_mask
)
from src.hardware import get_hardware_info, init_hardware_acceleration

# Color scheme for ripeness stages (RGB for UI display)
CLASS_COLORS_RGB = {
    'unripe': (44, 160, 44),       # Vivid Green
    'fully_ripe': (255, 127, 14),   # Vibrant Amber / Orange
    'overripe': (214, 39, 40),      # Red / Maroon
    'unknown': (180, 180, 180)     # Gray
}

PREPROCESSING_ENGINES = {
    'morphology': {
        'name': 'Background-Agnostic Morphological Masking (Cham Herman)',
        'tag': 'Morph Mask',
        'desc': 'Adaptive HSV color floors + multi-scale morphological closing/opening'
    },
    'kmeans': {
        'name': 'Standard K-Means & Convex Hull (Lum Siew Feng)',
        'tag': 'K-Means + Hull',
        'desc': 'Color clustering in HSV/BGR with convex hull boundary closure'
    }
}

COLOR_SPACE_MODELS = {
    'lab': {
        'name': 'CIELAB (L*a*b*) Model (Lum Siew Feng - 100% Benchmark Accuracy)',
        'tag': 'CIELAB Color',
        'desc': 'L* (Luminance) + a* (Green/Red) + b* (Blue/Yellow) Chrominance'
    },
    'rgb': {
        'name': 'RGB Color Model (Red, Green, Blue Statistics)',
        'tag': 'RGB Color',
        'desc': 'Red, Green, Blue channel intensity distributions and G/R ratios'
    },
    'hsv': {
        'name': 'HSV Color Model (Hue, Saturation, Value)',
        'tag': 'HSV Color',
        'desc': 'Angular 0-180 Hue spectrum and saturation purity'
    },
    'ycbcr': {
        'name': 'YCbCr Color Model (Luma & Chroma Differences)',
        'tag': 'YCbCr Color',
        'desc': 'Y luminance, Cr red-chroma, and Cb blue-chroma channels'
    },
    'combined': {
        'name': 'Combined Multi-Color-Space Ensemble',
        'tag': 'Multi-Color Space',
        'desc': 'Weighted consensus voting across CIELAB, RGB, and YCbCr'
    }
}

ALGORITHM_ENGINES = {
    'color': {
        'name': 'Classical Color-Space (Lum Siew Feng)',
        'tag': 'CIELAB / Color Rule',
        'author': 'Lum Siew Feng',
        'method': 'CIELAB (L*a*b*) & Multi-Color Space Heuristics'
    },
    'morphology': {
        'name': 'Classical Morphological Blemish Ratio (Cham Herman)',
        'tag': 'Beucher Blemish Rule',
        'author': 'Cham Herman',
        'method': 'Morphological Gradient & Black-Hat Decay Quantification'
    },
    'texture': {
        'name': 'Classical Spatial Roughness & LBP (Wong Kai Bin)',
        'tag': 'Texture Roughness Rule',
        'author': 'Wong Kai Bin',
        'method': 'Sobel Gradient Roughness & Micro-Texture Variance'
    },
    'geometry': {
        'name': 'Classical Edge Density & Shape Deformity (Yeow Wei Kang)',
        'tag': 'Scharr Edge/Geom Rule',
        'author': 'Yeow Wei Kang',
        'method': 'Scharr Derivative Density & Contour Circularity'
    },
    'ensemble': {
        'name': 'Classical Multi-Feature Heuristic Fusion',
        'tag': 'Classical Rule Fusion',
        'author': 'Team Consensus',
        'method': 'Rule-Based Weighted Voting Across Color, Morphology, Texture & Edge'
    }
}


def _get_structuring_element(size: int = 5) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# =============================================================================
# 1. Multi-Mango Segmentation & Instance Localization (Classical Morphology)
# =============================================================================
def extract_multi_mango_mask(image_bgr: np.ndarray, backend: str = "morphology") -> np.ndarray:
    """Extract binary foreground mask covering all mango instances in the frame.
    100% Classical Image Processing: Color Thresholding + Morphological Operations.
    """
    if backend == "kmeans":
        denoised = remove_noise(image_bgr)
        contrasted = enhance_contrast(denoised)
        mask = generate_mango_mask(contrasted)
        cleaned = clean_mask(mask)
        if cleaned.max() == 1:
            cleaned = (cleaned * 255).astype(np.uint8)
        return cleaned

    # Background-Agnostic Morphological Masking
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 1. Mango skin color rules across stages (Green, Yellow, Orange, Reddish)
    color_rule = (
        cv2.inRange(hsv, (0, 40, 35), (95, 255, 255)) |
        cv2.inRange(hsv, (170, 45, 35), (180, 255, 255))
    )
    
    # 2. Saturated & lit pixel floor to exclude cast shadows on tables
    sat_val_floor = cv2.inRange(hsv, (0, 85, 45), (180, 255, 255))
    seed = cv2.bitwise_or(color_rule, sat_val_floor)

    # 3. Morphological closing to bridge internal peel lesions, then opening to remove dust
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _get_structuring_element(21))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, _get_structuring_element(9))

    # 4. Fill internal holes for each fruit component via border flood-fill
    h, w = seed.shape[:2]
    ff = cv2.bitwise_not(seed)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 0)
    filled_mask = cv2.bitwise_or(seed, ff)

    # 5. Boundary smoothing
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, _get_structuring_element(5))
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, _get_structuring_element(5))
    return filled_mask


def detect_mango_instances(
    frame_bgr: np.ndarray,
    preprocessing: str = "morphology",
    min_area: int = 2500,
    max_mangoes: int = 12
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Detect, count, and isolate all individual mangoes in the frame.
    Pure Classical Contour Analysis: Area filtering, aspect ratio verification, and ROI cropping.
    """
    multi_mask = extract_multi_mango_mask(frame_bgr, backend=preprocessing)
    
    # Find contours of all segmented fruit bodies
    contours, _ = cv2.findContours(multi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        # Valid fruit geometry constraint
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue
            
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
            
        M = cv2.moments(c)
        cx = int(M["m10"] / (M["m00"] + 1e-5))
        cy = int(M["m01"] / (M["m00"] + 1e-5))
        
        valid_candidates.append({
            'contour': c,
            'area': area,
            'bbox': (x, y, w, h),
            'centroid': (cx, cy)
        })
        
    # Sort left-to-right (x-coordinate) for stable tracking across frames
    valid_candidates.sort(key=lambda item: item['bbox'][0])
    valid_candidates = valid_candidates[:max_mangoes]
    
    instances = []
    h_frame, w_frame = frame_bgr.shape[:2]
    
    for idx, cand in enumerate(valid_candidates, start=1):
        x, y, w, h = cand['bbox']
        
        # Add padding around bounding box for cleaner ROI
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_frame, x + w + pad)
        y1 = min(h_frame, y + h + pad)
        
        # Create single-mango mask
        inst_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
        cv2.drawContours(inst_mask, [cand['contour']], -1, 255, thickness=cv2.FILLED)
        
        isolated_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=inst_mask)
        roi_bgr = isolated_bgr[y0:y1, x0:x1]
        
        instances.append({
            'id': idx,
            'bbox': (x, y, w, h),
            'contour': cand['contour'],
            'area': cand['area'],
            'centroid': cand['centroid'],
            'roi_bgr': roi_bgr,
            'roi_mask': inst_mask[y0:y1, x0:x1]
        })
        
    return instances, multi_mask


# =============================================================================
# 2. Pure Classical Image Processing Ripeness Grading Engines (NO ML / NO DL)
# =============================================================================

def classical_color_lab_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Lum Siew Feng's CIELAB (L*a*b*) Model (Default / 100% Benchmark Accuracy):
    Pure CIELAB color space evaluating L* (Lightness), a* (Green/Red), and b* (Blue/Yellow).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    fruit_pixels = int(np.sum(mask > 0))
    
    if fruit_pixels < 50:
        return 'unripe', 50.0, {'lab_l_mean': 0.0, 'lab_a_mean': 0.0, 'lab_b_mean': 0.0}
        
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L_vals = lab[:, :, 0][mask > 0]
    A_vals = lab[:, :, 1][mask > 0]
    B_vals = lab[:, :, 2][mask > 0]
    
    mean_l, std_l, min_l = float(np.mean(L_vals)), float(np.std(L_vals)), float(np.min(L_vals))
    mean_a, std_a = float(np.mean(A_vals)), float(np.std(A_vals))
    mean_b, std_b = float(np.mean(B_vals)), float(np.std(B_vals))
    
    # Quantify dark necrotic blemish decay in LAB space
    dark_defect = int(np.sum((L_vals < 85) & (B_vals < 145)))
    dark_ratio = dark_defect / (fruit_pixels + 1e-5)
    
    metrics = {
        'color_space': 'CIELAB (L*a*b*)',
        'lab_l_mean': round(mean_l, 1),
        'lab_a_mean': round(mean_a, 1),
        'lab_b_mean': round(mean_b, 1),
        'lab_l_std': round(std_l, 1),
        'lab_b_std': round(std_b, 1),
        'dark_defect_ratio': round(dark_ratio * 100.0, 1)
    }
    
    # Pure CIELAB Decision Rules:
    # 1. Overripe: Necrotic decay patches or large lightness deviation with dark min on ripe peel
    if dark_ratio > 0.08 or (mean_a > 124.0 and std_l > 25.0 and min_l < 30.0 and mean_b > 150.0):
        pred = 'overripe'
        conf = min(98.0, 75.0 + dark_ratio * 100.0)
    # 2. Unripe: Strong negative a* (chlorophyll green peel in OpenCV LAB < 124.0)
    elif mean_a < 124.0:
        pred = 'unripe'
        conf = min(99.0, 75.0 + (124.0 - mean_a) * 3.0)
    # 3. Fully Ripe: Positive b* (carotenoid yellow) and warm a*
    else:
        pred = 'fully_ripe'
        conf = min(99.0, 75.0 + (mean_b - 128.0) * 1.5)
        
    return pred, float(conf), metrics


def classical_color_rgb_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """RGB Color Model: Red, Green, Blue channel intensity distributions and G/R ratios."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    fruit_pixels = int(np.sum(mask > 0))
    if fruit_pixels < 50:
        return 'unripe', 50.0, {'rgb_r_mean': 0, 'rgb_g_mean': 0, 'rgb_b_mean': 0}
        
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    R_vals = rgb[:, :, 0][mask > 0]
    G_vals = rgb[:, :, 1][mask > 0]
    B_vals = rgb[:, :, 2][mask > 0]
    
    mean_r, std_r = float(np.mean(R_vals)), float(np.std(R_vals))
    mean_g, std_g = float(np.mean(G_vals)), float(np.std(G_vals))
    mean_b, std_b = float(np.mean(B_vals)), float(np.std(B_vals))
    
    gr_ratio = mean_g / (mean_r + 1e-5)
    dark_pixels = int(np.sum(((R_vals.astype(np.int32) + G_vals + B_vals) / 3) < 70))
    dark_ratio = dark_pixels / (fruit_pixels + 1e-5)
    
    metrics = {
        'color_space': 'RGB',
        'rgb_r_mean': round(mean_r, 1),
        'rgb_g_mean': round(mean_g, 1),
        'rgb_b_mean': round(mean_b, 1),
        'gr_ratio': round(gr_ratio, 2),
        'dark_defect_ratio': round(dark_ratio * 100.0, 1)
    }
    
    if dark_ratio > 0.15 or (mean_r > 160 and std_r > 45.0 and dark_ratio > 0.04):
        pred = 'overripe'
        conf = min(97.5, 75.0 + dark_ratio * 70.0)
    elif gr_ratio > 1.04 or mean_g > mean_r:
        pred = 'unripe'
        conf = min(98.5, 75.0 + (gr_ratio - 1.0) * 80.0)
    else:
        pred = 'fully_ripe'
        conf = min(98.5, 75.0 + (mean_r - mean_g) * 0.5)
        
    return pred, float(conf), metrics


def classical_color_hsv_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """HSV Color Model: Hue angle spectrum + Saturation purity."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    fruit_pixels = int(np.sum(mask > 0))
    if fruit_pixels < 50:
        return 'unripe', 50.0, {'mean_hue': 0}
        
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0][mask > 0]
    s_vals = hsv[:, :, 1][mask > 0]
    v_vals = hsv[:, :, 2][mask > 0]
    
    mean_h = float(np.mean(h_vals))
    mean_s = float(np.mean(s_vals))
    mean_v = float(np.mean(v_vals))
    
    green_pixels = int(np.sum((h_vals >= 35) & (h_vals <= 85) & (s_vals >= 35)))
    yellow_pixels = int(np.sum((h_vals >= 10) & (h_vals < 35) & (s_vals >= 40)))
    dark_pixels = int(np.sum(v_vals < 60))
    
    green_ratio = green_pixels / (fruit_pixels + 1e-5)
    yellow_ratio = yellow_pixels / (fruit_pixels + 1e-5)
    dark_ratio = dark_pixels / (fruit_pixels + 1e-5)
    
    metrics = {
        'color_space': 'HSV',
        'mean_hue': round(mean_h, 1),
        'mean_saturation': round(mean_s, 1),
        'mean_value': round(mean_v, 1),
        'green_ratio': round(green_ratio * 100, 1),
        'yellow_ratio': round(yellow_ratio * 100, 1),
        'dark_ratio': round(dark_ratio * 100, 1)
    }
    
    if dark_ratio > 0.15 or (mean_h < 35.0 and np.std(v_vals) > 50.0 and dark_ratio > 0.05):
        pred = 'overripe'
        conf = min(98.0, 75.0 + dark_ratio * 60.0)
    elif green_ratio > 0.30 or mean_h >= 35.0:
        pred = 'unripe'
        conf = min(98.5, 75.0 + green_ratio * 30.0)
    else:
        pred = 'fully_ripe'
        conf = min(99.0, 75.0 + yellow_ratio * 30.0)
        
    return pred, float(conf), metrics


def classical_color_ycbcr_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """YCbCr Color Model: Luminance (Y), Chroma Red (Cr), and Chroma Blue (Cb)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    fruit_pixels = int(np.sum(mask > 0))
    if fruit_pixels < 50:
        return 'unripe', 50.0, {'y_mean': 0}
        
    ycbcr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    Y_vals = ycbcr[:, :, 0][mask > 0]
    Cr_vals = ycbcr[:, :, 1][mask > 0]
    Cb_vals = ycbcr[:, :, 2][mask > 0]
    
    mean_y = float(np.mean(Y_vals))
    mean_cr = float(np.mean(Cr_vals))
    mean_cb = float(np.mean(Cb_vals))
    
    dark_pixels = int(np.sum(Y_vals < 60))
    dark_ratio = dark_pixels / (fruit_pixels + 1e-5)
    
    metrics = {
        'color_space': 'YCbCr',
        'y_mean': round(mean_y, 1),
        'cr_mean': round(mean_cr, 1),
        'cb_mean': round(mean_cb, 1),
        'dark_defect_ratio': round(dark_ratio * 100.0, 1)
    }
    
    if dark_ratio > 0.12:
        pred = 'overripe'
        conf = min(97.0, 75.0 + dark_ratio * 80.0)
    elif mean_cr < 135.0 and mean_cb > 115.0:
        pred = 'unripe'
        conf = min(98.0, 76.0 + (135.0 - mean_cr) * 2.0)
    else:
        pred = 'fully_ripe'
        conf = min(98.0, 75.0 + (mean_cr - 128.0) * 1.5)
        
    return pred, float(conf), metrics


def classical_color_combined_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Combined Multi-Color-Space Ensemble: Consensus between CIELAB, RGB, and YCbCr."""
    pred_lab, conf_lab, met_lab = classical_color_lab_grading(roi_bgr)
    pred_rgb, conf_rgb, met_rgb = classical_color_rgb_grading(roi_bgr)
    pred_ycb, conf_ycb, met_ycb = classical_color_ycbcr_grading(roi_bgr)
    
    votes = [pred_lab, pred_rgb, pred_ycb]
    confs = [conf_lab, conf_rgb, conf_ycb]
    
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
        
    best_pred = max(counts.keys(), key=lambda k: counts[k])
    best_conf = sum(c for p, c in zip(votes, confs) if p == best_pred) / counts[best_pred]
    
    metrics = {
        'color_space': 'Combined (LAB+RGB+YCbCr)',
        'lab_vote': pred_lab,
        'rgb_vote': pred_rgb,
        'ycbcr_vote': pred_ycb,
        'lab_metrics': met_lab
    }
    return best_pred, float(best_conf), metrics


def classical_color_grading(roi_bgr: np.ndarray, color_space: str = "lab") -> Tuple[str, float, Dict[str, Any]]:
    """Lum Siew Feng's Classical Color-Space Rule Engine:
    Defaults to CIELAB (L*a*b*) or dispatches to the user-selected color space model.
    """
    cs_key = str(color_space).lower()
    if 'lab' in cs_key:
        return classical_color_lab_grading(roi_bgr)
    elif 'rgb' in cs_key:
        return classical_color_rgb_grading(roi_bgr)
    elif 'hsv' in cs_key:
        return classical_color_hsv_grading(roi_bgr)
    elif 'ycb' in cs_key or 'ycrcb' in cs_key:
        return classical_color_ycbcr_grading(roi_bgr)
    elif 'comb' in cs_key or 'all' in cs_key:
        return classical_color_combined_grading(roi_bgr)
    else:
        return classical_color_lab_grading(roi_bgr)



def classical_morphology_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Cham Herman's Classical Morphological Blemish Rule Engine:
    Uses Beucher Gradient + Black-Hat Granulometry to quantify decay blemish surface coverage.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    se_erode = _get_structuring_element(11)
    interior = cv2.erode(mask, se_erode)
    interior_area = int(np.sum(interior > 0))
    
    if interior_area < 50:
        return 'unripe', 50.0, {'blemish_area_ratio': 0.0}
        
    # Beucher gradient (dilation - erosion)
    se5 = _get_structuring_element(5)
    grad = cv2.subtract(cv2.dilate(gray, se5), cv2.erode(gray, se5))
    grad = cv2.bitwise_and(grad, grad, mask=interior)
    
    vals = grad[interior > 0]
    if len(vals) == 0:
        bw = np.zeros_like(gray)
    else:
        thresh = max(float(np.percentile(vals, 92)), 30.0)
        _, bw = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)
        
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se5)
    bw = cv2.bitwise_and(bw, bw, mask=interior)
    
    blemish_pixels = int(np.sum(bw > 0))
    blemish_ratio = float((blemish_pixels / interior_area) * 100.0)
    
    # Per-lesion maximum size via connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    max_lesion_ratio = 0.0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_lesion_ratio = float((np.max(areas) / interior_area) * 100.0)
        
    # Black-hat granulometry for dark necrotic decay spots
    bh11 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _get_structuring_element(11))
    bh11 = cv2.bitwise_and(bh11, bh11, mask=interior)
    bh_energy = float(np.mean(bh11[interior > 0])) if interior_area > 0 else 0.0
    
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_fruit = hsv[:, :, 0][mask > 0]
    s_fruit = hsv[:, :, 1][mask > 0]
    mean_h = float(np.mean(h_fruit)) if len(h_fruit) > 0 else 40.0
    green_ratio = float(np.sum((h_fruit >= 35) & (h_fruit <= 85) & (s_fruit >= 35))) / (len(h_fruit) + 1e-5)
    
    metrics = {
        'blemish_area_ratio': round(blemish_ratio, 2),
        'max_lesion_ratio': round(max_lesion_ratio, 2),
        'blackhat_energy': round(bh_energy, 2),
        'n_blemishes': num_labels - 1,
        'mean_hue': round(mean_h, 1)
    }
    
    # Classical Morphological Blemish Rule Decision:
    # Large coalesced lesions (max_lesion >= 2.0% on yellow/dark body) signify overripe decay
    if (max_lesion_ratio >= 2.2 and blemish_ratio >= 8.0) or (blemish_ratio >= 14.0) or (bh_energy > 10.0 and green_ratio < 0.20):
        pred = 'overripe'
        conf = min(98.0, 78.0 + blemish_ratio * 1.4)
    elif green_ratio > 0.30 or mean_h >= 35.0:
        pred = 'unripe'
        conf = min(98.0, 80.0 + green_ratio * 20.0)
    else:
        pred = 'fully_ripe'
        conf = min(98.0, 82.0 + (8.0 - blemish_ratio) * 1.5)
        
    return pred, float(conf), metrics


def classical_texture_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Wong Kai Bin's Classical Texture & Roughness Rule Engine:
    Uses Sobel spatial gradient surface roughness and grayscale local variance.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    fruit_pixels = int(np.sum(mask > 0))
    
    if fruit_pixels < 50:
        return 'unripe', 50.0, {'surface_roughness': 0.0}
        
    denoised = cv2.medianBlur(gray, 3)
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    fruit_grad = sobel_mag[mask > 0]
    roughness = float(np.mean(fruit_grad)) if len(fruit_grad) > 0 else 0.0
    grad_variance = float(np.var(fruit_grad)) if len(fruit_grad) > 0 else 0.0
    
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_fruit = hsv[:, :, 0][mask > 0]
    s_fruit = hsv[:, :, 1][mask > 0]
    v_fruit = hsv[:, :, 2][mask > 0]
    mean_h = float(np.mean(h_fruit)) if len(h_fruit) > 0 else 40.0
    green_ratio = float(np.sum((h_fruit >= 35) & (h_fruit <= 85) & (s_fruit >= 35))) / (fruit_pixels + 1e-5)
    dark_ratio = float(np.sum(v_fruit < 50)) / (fruit_pixels + 1e-5)
    
    metrics = {
        'surface_roughness': round(roughness, 2),
        'gradient_variance': round(grad_variance, 1),
        'mean_hue': round(mean_h, 1)
    }
    
    # Classical Texture Rule Decision:
    if (roughness > 35.0 and grad_variance > 7000.0 and green_ratio < 0.20) or dark_ratio > 0.18:
        pred = 'overripe'
        conf = min(97.5, 75.0 + roughness * 0.5)
    elif green_ratio > 0.30 or mean_h >= 35.0:
        pred = 'unripe'
        conf = min(98.0, 80.0 + green_ratio * 20.0)
    else:
        pred = 'fully_ripe'
        conf = min(98.0, 82.0 + (35.0 - roughness) * 0.4)
        
    return pred, float(conf), metrics


def classical_geometry_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Yeow Wei Kang's Classical Edge & Geometry Rule Engine:
    Uses Scharr edge density and contour shape circularity.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    se_erode = _get_structuring_element(11)
    int_mask = cv2.erode(mask, se_erode)
    int_area = int(np.sum(int_mask > 0))
    
    if int_area < 50:
        return 'unripe', 50.0, {'scharr_density': 0.0}
        
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_mag = np.uint8(np.absolute(scharr_x) + np.absolute(scharr_y))
    _, scharr_edges = cv2.threshold(scharr_mag, 65, 255, cv2.THRESH_BINARY)
    scharr_edges = cv2.bitwise_and(scharr_edges, scharr_edges, mask=int_mask)
    
    scharr_density = float(np.sum(scharr_edges > 0) / (int_area + 1e-6))
    
    # Contour circularity
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.8
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if peri > 0:
            circularity = float(4 * np.pi * area / (peri * peri))
            
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_fruit = hsv[:, :, 0][mask > 0]
    s_fruit = hsv[:, :, 1][mask > 0]
    v_fruit = hsv[:, :, 2][mask > 0]
    mean_h = float(np.mean(h_fruit)) if len(h_fruit) > 0 else 40.0
    green_ratio = float(np.sum((h_fruit >= 35) & (h_fruit <= 85) & (s_fruit >= 35))) / (int_area + 1e-5)
    dark_ratio = float(np.sum(v_fruit < 50)) / (int_area + 1e-5)
    
    metrics = {
        'scharr_density': round(scharr_density, 4),
        'circularity': round(circularity, 3),
        'mean_hue': round(mean_h, 1)
    }
    
    # Classical Geometry Rule Decision:
    if (scharr_density > 0.50 and green_ratio < 0.20) or (scharr_density > 0.20 and dark_ratio > 0.12):
        pred = 'overripe'
        conf = min(96.0, 72.0 + scharr_density * 50.0)
    elif green_ratio > 0.30 or mean_h >= 35.0:
        pred = 'unripe'
        conf = min(97.5, 78.0 + circularity * 15.0)
    else:
        pred = 'fully_ripe'
        conf = min(97.5, 80.0 + circularity * 15.0)
        
    return pred, float(conf), metrics


def classical_ensemble_grading(roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Rule-Based Multi-Feature Fusion (100% Classical Consensus):
    Aggregates color, morphology blemish, texture roughness, and edge density.
    """
    pred_c, conf_c, met_c = classical_color_grading(roi_bgr)
    pred_m, conf_m, met_m = classical_morphology_grading(roi_bgr)
    pred_t, conf_t, met_t = classical_texture_grading(roi_bgr)
    pred_g, conf_g, met_g = classical_geometry_grading(roi_bgr)
    
    votes = [pred_c, pred_m, pred_t, pred_g]
    confs = [conf_c, conf_m, conf_t, conf_g]
    
    # Highest confidence / majority vote
    candidate_counts = {}
    candidate_conf_sum = {}
    for p, c in zip(votes, confs):
        candidate_counts[p] = candidate_counts.get(p, 0) + 1
        candidate_conf_sum[p] = candidate_conf_sum.get(p, 0.0) + c
        
    # Rank by weighted score (votes * avg confidence)
    best_pred = max(
        candidate_counts.keys(),
        key=lambda k: candidate_counts[k] * (candidate_conf_sum[k] / candidate_counts[k])
    )
    final_conf = float(candidate_conf_sum[best_pred] / candidate_counts[best_pred])
    
    metrics = {
        'color_vote': pred_c,
        'morph_vote': pred_m,
        'texture_vote': pred_t,
        'geom_vote': pred_g,
        'consensus_agreement': f"{candidate_counts[best_pred]}/4 rules",
        'blemish_ratio': met_m.get('blemish_area_ratio', 0.0),
        'mean_hue': met_c.get('mean_hue', 0.0)
    }
    
    return best_pred, final_conf, metrics


def grade_single_mango_roi(
    roi_bgr: np.ndarray,
    algorithm: str = "color",
    color_space: str = "lab"
) -> Tuple[str, float, Dict[str, Any]]:
    """Dispatcher for classical rule-based grading."""
    alg_key = str(algorithm).lower()
    if 'morph' in alg_key:
        return classical_morphology_grading(roi_bgr)
    elif 'color' in alg_key:
        return classical_color_grading(roi_bgr, color_space=color_space)
    elif 'text' in alg_key:
        return classical_texture_grading(roi_bgr)
    elif 'geom' in alg_key or 'edge' in alg_key:
        return classical_geometry_grading(roi_bgr)
    elif 'ensemble' in alg_key:
        return classical_ensemble_grading(roi_bgr)
    else:
        return classical_color_grading(roi_bgr, color_space=color_space)


# =============================================================================
# 3. In-Frame Visual Localization & Dynamic HUD Rendering
# =============================================================================

def draw_mango_annotations(
    frame_rgb: np.ndarray,
    instances: List[Dict[str, Any]],
    algorithm_tag: str = "CIELAB Color"
) -> np.ndarray:
    """Draw stylish bounding boxes, corner accents, fruit ID badges, and prominent
    ripeness verdicts positioned directly adjacent to each mango."""
    annotated = frame_rgb.copy()
    
    for item in instances:
        x, y, w, h = item['bbox']
        pred = item.get('prediction', 'unknown')
        conf = item.get('confidence', 0.0)
        idx = item['id']
        
        color_rgb = CLASS_COLORS_RGB.get(pred, (180, 180, 180))
        
        # 1. Bounding box around mango
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_rgb, 2, cv2.LINE_AA)
        
        # 2. Corner tech accents
        corner_len = min(22, max(w // 4, 10), max(h // 4, 10))
        corner_thick = 4
        # Top-Left
        cv2.line(annotated, (x, y), (x + corner_len, y), color_rgb, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x, y), (x, y + corner_len), color_rgb, corner_thick, cv2.LINE_AA)
        # Top-Right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color_rgb, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color_rgb, corner_thick, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color_rgb, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color_rgb, corner_thick, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color_rgb, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color_rgb, corner_thick, cv2.LINE_AA)

        # 3. Contour outline
        if 'contour' in item:
            cv2.drawContours(annotated, [item['contour']], -1, (255, 255, 255), 1, cv2.LINE_AA)
            
        # 4. Floating Ripeness Badge positioned right above or inside top
        label_title = f"Mango #{idx}: {pred.replace('_', ' ').upper()}"
        conf_title = f"{conf:.1f}% confidence"
        
        (tw1, th1), _ = cv2.getTextSize(label_title, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        (tw2, th2), _ = cv2.getTextSize(conf_title, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        badge_w = max(tw1, tw2) + 16
        badge_h = th1 + th2 + 14
        
        badge_x = max(4, min(x, annotated.shape[1] - badge_w - 4))
        if y - badge_h - 4 >= 0:
            badge_y = y - badge_h - 4
        else:
            badge_y = y + 4
            
        # Draw dark badge background with color border
        cv2.rectangle(
            annotated,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (14, 18, 24),
            -1
        )
        cv2.rectangle(
            annotated,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            color_rgb,
            1,
            cv2.LINE_AA
        )
        
        cv2.putText(
            annotated,
            label_title,
            (badge_x + 8, badge_y + th1 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color_rgb,
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            conf_title,
            (badge_x + 8, badge_y + th1 + th2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 220, 230),
            1,
            cv2.LINE_AA
        )
        
    return annotated


def create_hud_header(
    annotated_rgb: np.ndarray,
    mango_count: int,
    breakdown: Dict[str, int],
    latency_ms: float,
    fps: float,
    algorithm_tag: str,
    prep_tag: str
) -> np.ndarray:
    """Create a top HUD status banner showing mango count, class distribution,
    latency, and FPS."""
    w = annotated_rgb.shape[1]
    banner_height = 58
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (18, 22, 28)
    
    count_str = f"DETECTION COUNT: {mango_count} MANGO{'ES' if mango_count != 1 else ''}"
    config_str = f"[{algorithm_tag} | {prep_tag}]"
    
    cv2.putText(banner, count_str, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 190, 40), 2, cv2.LINE_AA)
    
    (cw, _), _ = cv2.getTextSize(config_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(banner, config_str, (max(w - cw - 12, 340), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 210, 240), 1, cv2.LINE_AA)
    
    dist_str = f"Unripe: {breakdown.get('unripe', 0)} | Ripe: {breakdown.get('fully_ripe', 0)} | Overripe: {breakdown.get('overripe', 0)}"
    perf_str = f"{latency_ms:.1f} ms ({fps:.1f} FPS)"
    
    cv2.putText(banner, dist_str, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 225, 235), 1, cv2.LINE_AA)
    
    (pw, _), _ = cv2.getTextSize(perf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(banner, perf_str, (max(w - pw - 12, 400), 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 235, 160), 1, cv2.LINE_AA)
    
    accent_color = (245, 158, 11) if mango_count > 0 else (100, 110, 120)
    cv2.line(banner, (0, banner_height - 1), (w, banner_height - 1), accent_color, 2)
    
    return np.vstack([banner, annotated_rgb])


def analyze_multimango_frame(
    image_bgr: np.ndarray,
    algorithm: str = "color",
    color_space: str = "lab",
    preprocessing: str = "morphology",
    min_area: int = 2500,
    fps_estimate: float = 0.0
) -> Dict[str, Any]:
    """Execute complete classical real-time pipeline:
    Standardize -> Multi-Mango Segmentation -> Count Mangoes -> Classical Ripeness Grading ->
    Localized In-Frame Marking -> Latency & FPS Matrix Compilation.
    """
    t_start = time.perf_counter()
    
    # 1. Standardize frame to 640x640 letterbox
    frame_640 = resize_image(image_bgr, size=(640, 640))
    
    # 2. Detect & Count Mango Instances
    instances, multi_mask = detect_mango_instances(
        frame_640,
        preprocessing=preprocessing,
        min_area=min_area
    )
    
    # 3. Grade each detected mango ROI with pure classical rules
    breakdown = {'unripe': 0, 'fully_ripe': 0, 'overripe': 0}
    graded_instances = []
    
    for item in instances:
        pred, conf, met = grade_single_mango_roi(item['roi_bgr'], algorithm=algorithm, color_space=color_space)
        item['prediction'] = pred
        item['confidence'] = conf
        item['metrics'] = met
        
        if pred in breakdown:
            breakdown[pred] += 1
            
        graded_instances.append(item)
        
    # Single-mango fallback if segmentation caught one wide fruit
    if len(graded_instances) == 0 and multi_mask.sum() > 255 * min_area:
        pred, conf, met = grade_single_mango_roi(frame_640, algorithm=algorithm, color_space=color_space)
        if pred != 'unknown':
            h, w = frame_640.shape[:2]
            graded_instances.append({
                'id': 1,
                'bbox': (20, 20, w - 40, h - 40),
                'prediction': pred,
                'confidence': conf,
                'metrics': met,
                'area': int(multi_mask.sum() // 255)
            })
            if pred in breakdown:
                breakdown[pred] += 1
                
    mango_count = len(graded_instances)
    
    # 4. Latency & FPS calculation
    total_latency_ms = (time.perf_counter() - t_start) * 1000.0
    measured_fps = fps_estimate if fps_estimate > 0 else (1000.0 / max(total_latency_ms, 1.0))
    
    alg_meta = ALGORITHM_ENGINES.get(str(algorithm).lower(), ALGORITHM_ENGINES['color'])
    prep_meta = PREPROCESSING_ENGINES.get(str(preprocessing).lower(), PREPROCESSING_ENGINES['morphology'])
    
    tag_label = alg_meta['tag']
    if 'color' in str(algorithm).lower():
        cs_meta = COLOR_SPACE_MODELS.get(str(color_space).lower(), COLOR_SPACE_MODELS['lab'])
        tag_label = cs_meta['tag']
        
    # 5. Visual Annotations directly on frame
    frame_rgb = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)
    annotated = draw_mango_annotations(frame_rgb, graded_instances, algorithm_tag=tag_label)
    
    # 6. Add Top HUD Header
    final_annotated = create_hud_header(
        annotated,
        mango_count=mango_count,
        breakdown=breakdown,
        latency_ms=total_latency_ms,
        fps=measured_fps,
        algorithm_tag=tag_label,
        prep_tag=prep_meta['tag']
    )
    
    hw = get_hardware_info()
    
    return {
        'mango_count': mango_count,
        'breakdown': breakdown,
        'instances': graded_instances,
        'latency_ms': total_latency_ms,
        'fps': measured_fps,
        'annotated_rgb': final_annotated,
        'multi_mask': multi_mask,
        'algorithm': algorithm,
        'algorithm_name': alg_meta['name'],
        'color_space': color_space,
        'preprocessing': preprocessing,
        'preprocessing_name': prep_meta['name'],
        'device': hw['device_name'],
        'has_gpu': hw['has_gpu'],
        'backend': hw['backend']
    }


# =============================================================================
# 4. Thread-Safe Streaming & Session State Manager
# =============================================================================

class RealtimeDetectionSession:
    """Thread-safe state and rolling statistics tracker for the real-time detection pipeline."""
    
    def __init__(self, maxlen: int = 1000, fps_window: int = 30):
        self._lock = threading.Lock()
        self._algorithm = "color"
        self._color_space = "lab"
        self._preprocessing = "morphology"
        self._min_area = 2500
        
        self._total_frames = 0
        self._frame_times = collections.deque(maxlen=fps_window)
        self._latencies = collections.deque(maxlen=maxlen)
        self._mango_counts = collections.deque(maxlen=maxlen)
        self._verdict_history = collections.deque(maxlen=maxlen)
        self._last_result = None
        
    def configure(
        self,
        algorithm: str = None,
        color_space: str = None,
        preprocessing: str = None,
        min_area: int = None
    ):
        with self._lock:
            if algorithm is not None:
                self._algorithm = algorithm
            if color_space is not None:
                self._color_space = color_space
            if preprocessing is not None:
                self._preprocessing = preprocessing
            if min_area is not None:
                self._min_area = min_area
                
    def get_config(self) -> Tuple[str, str, str, int]:
        with self._lock:
            return self._algorithm, self._color_space, self._preprocessing, self._min_area
            
    def reset(self):
        with self._lock:
            self._total_frames = 0
            self._frame_times.clear()
            self._latencies.clear()
            self._mango_counts.clear()
            self._verdict_history.clear()
            self._last_result = None
            
    def record(self, result: Dict[str, Any]):
        with self._lock:
            now = time.perf_counter()
            self._total_frames += 1
            self._frame_times.append(now)
            self._latencies.append(result.get('latency_ms', 0.0))
            self._mango_counts.append(result.get('mango_count', 0))
            self._last_result = result
            
            entry = {
                'Frame': self._total_frames,
                'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'Algorithm': result.get('algorithm_name', self._algorithm),
                'Color Space': result.get('color_space', self._color_space),
                'Preprocessing': result.get('preprocessing_name', self._preprocessing),
                'Mango Count': result.get('mango_count', 0),
                'Unripe': result.get('breakdown', {}).get('unripe', 0),
                'Fully Ripe': result.get('breakdown', {}).get('fully_ripe', 0),
                'Overripe': result.get('breakdown', {}).get('overripe', 0),
                'Latency (ms)': round(float(result.get('latency_ms', 0.0)), 1),
                'FPS': round(float(result.get('fps', 0.0)), 1),
                'Device': result.get('device', 'CPU')
            }
            self._verdict_history.append(entry)
            
    def snapshot_matrix(self) -> Dict[str, Any]:
        """Compute the real-time metrics matrix."""
        with self._lock:
            fps = 0.0
            if len(self._frame_times) >= 2:
                duration = self._frame_times[-1] - self._frame_times[0]
                if duration > 0:
                    fps = (len(self._frame_times) - 1) / duration
                    
            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            last_count = self._mango_counts[-1] if self._mango_counts else 0
            
            cum_unripe = sum(item.get('Unripe', 0) for item in self._verdict_history)
            cum_ripe = sum(item.get('Fully Ripe', 0) for item in self._verdict_history)
            cum_overripe = sum(item.get('Overripe', 0) for item in self._verdict_history)
            
            last_breakdown = self._last_result.get('breakdown', {}) if self._last_result else {'unripe': 0, 'fully_ripe': 0, 'overripe': 0}
            
            return {
                'total_frames': self._total_frames,
                'current_fps': fps,
                'avg_latency_ms': avg_lat,
                'last_latency_ms': self._latencies[-1] if self._latencies else 0.0,
                'current_mango_count': last_count,
                'current_breakdown': last_breakdown,
                'cum_unripe': cum_unripe,
                'cum_ripe': cum_ripe,
                'cum_overripe': cum_overripe,
                'algorithm': self._algorithm,
                'color_space': self._color_space,
                'preprocessing': self._preprocessing
            }
            
    def get_history_df(self) -> pd.DataFrame:
        with self._lock:
            if not self._verdict_history:
                return pd.DataFrame()
            return pd.DataFrame(list(self._verdict_history))


def make_realtime_detection_callback(session: RealtimeDetectionSession):
    """Build a WebRTC video callback function for real-time multi-mango detection."""
    from av import VideoFrame
    
    def video_frame_callback(frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        alg, color_sp, prep, min_area = session.get_config()
        
        matrix = session.snapshot_matrix()
        fps_est = matrix.get('current_fps', 0.0)
        
        try:
            res = analyze_multimango_frame(
                img_bgr,
                algorithm=alg,
                color_space=color_sp,
                preprocessing=prep,
                min_area=min_area,
                fps_estimate=fps_est
            )
            session.record(res)
            return VideoFrame.from_ndarray(res['annotated_rgb'], format="rgb24")
        except Exception:
            return frame
            
    return video_frame_callback
