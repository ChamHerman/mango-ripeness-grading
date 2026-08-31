import os
import time
import functools
import cv2
import numpy as np
import pandas as pd
import joblib

@functools.lru_cache(maxsize=4)
def _load_model_package(model_path):
    """Cache the serialized model package so per-frame video calls do not
    re-read the joblib file on every invocation."""
    return joblib.load(model_path)

def get_fruit_mask(img_bgr, thresh=20):
    """Extract and clean foreground mango mask from black background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thresh).astype(np.uint8) * 255
    se_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_clean)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_clean)
    return mask

def get_interior_mask(mask, k=13):
    """""Erode mango mask to eliminate boundary transition ring."""""
    se_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, se_erode)

def _ellipse(size):
    """Elliptical structuring element helper."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def run_enhanced_blemish_mask(gray, interior):
    """
    Primary blemish mask: Beucher Gradient Delineation.

    The proven Algorithm 3 backbone: dilation - erosion with a 5x5 elliptical
    structuring element, adaptive 92nd-percentile thresholding, and a closing
    pass to consolidate lesion perimeters.
    """
    se5 = _ellipse(5)
    grad = cv2.subtract(cv2.dilate(gray, se5), cv2.erode(gray, se5))
    grad = cv2.bitwise_and(grad, grad, mask=interior)

    vals = grad[interior > 0]
    if len(vals) == 0 or vals.max() == 0:
        bw = np.zeros_like(gray)
    else:
        thresh_val = max(np.percentile(vals, 92), 25)
        _, bw = cv2.threshold(grad, thresh_val, 255, cv2.THRESH_BINARY)

    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se5)
    bw = cv2.bitwise_and(bw, bw, mask=interior)
    return bw, {'grad': grad}

def extract_morphological_features(gray, bw_blemish, interior_mask, interior_area):
    """Extract standardised 10-dimensional base morphological feature vector."""
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

def extract_granulometry_features(gray, interior):
    """
    Granulometric pattern spectrum (pure morphology).

    Black-hat energies across increasing structuring-element sizes capture the
    size distribution of dark lesions (small lenticels vs coalesced anthracnose
    decay); the top-hat energy captures pale necrotic patches.
    """
    m = interior > 0
    feats = {}

    bh_energies = []
    for s in (7, 11, 15, 19):
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _ellipse(s))
        bh_energies.append(float(np.mean(bh[m]) / 255.0))
    feats['gran_small'] = bh_energies[0]
    feats['gran_medium'] = max(bh_energies[1] - bh_energies[0], 0.0)
    feats['gran_large'] = max(bh_energies[2] - bh_energies[1], 0.0)
    feats['gran_coalesced'] = max(bh_energies[3] - bh_energies[2], 0.0)

    bh11 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _ellipse(11))
    bh11 = cv2.bitwise_and(bh11, bh11, mask=interior)
    feats['dark_lesion_peak'] = float(np.percentile(bh11[m], 95)) if m.any() else 0.0

    th11 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _ellipse(11))
    feats['pale_patch_energy'] = float(np.mean(th11[m]) / 255.0)
    return feats

def count_split_lesions(bw_mask, interior_area, frac=0.01, return_vis=False):
    """
    Marker-controlled watershed split of large merged components (MRMF).

    Algorithm 4 mechanics repositioned inside the pipeline: distance-transform
    seeds separate touching lesions within large decay clusters.
    Returns (n_large_components, n_lesions_after_split) or, with return_vis=True,
    (n_large_components, n_lesions_after_split, vis) where vis highlights the
    watershed boundaries in yellow on the blemish mask.
    """
    vis = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2RGB) if return_vis else None
    n_large = 0
    split_total = 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_mask, connectivity=8)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= frac * interior_area:
            n_large += 1
            comp = (labels == i).astype(np.uint8) * 255
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            split = False
            if dist.max() > 2.0:
                _, seeds = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
                nm, markers = cv2.connectedComponents(seeds.astype(np.uint8))
                if nm > 1:
                    markers = markers + 1
                    markers[comp == 0] = 0
                    markers = cv2.watershed(cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR), markers)
                    k = len([v for v in np.unique(markers) if v > 1])
                    split_total += max(k, 1)
                    if return_vis:
                        vis[markers == -1] = (255, 255, 0)
                    split = True
            if not split:
                split_total += 1
    if return_vis:
        return n_large, split_total, vis
    return n_large, split_total

def extract_per_lesion_features(bw_mask, interior_area):
    """Per-lesion decay structure: largest lesion size, consolidated decay share,
    large-lesion count, and watershed-split true lesion count (MRMF)."""
    f = {'max_lesion_ratio': 0.0, 'consolidated_ratio': 0.0,
         'n_large_lesions': 0, 'n_lesions_split': 0}
    if interior_area == 0:
        return f
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_mask, connectivity=8)
    if num <= 1:
        return f
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    f['max_lesion_ratio'] = float(areas.max() / interior_area * 100.0)
    big = areas[areas >= 0.01 * interior_area]
    f['consolidated_ratio'] = float(big.sum() / interior_area * 100.0) if len(big) else 0.0
    n_large, split_total = count_split_lesions(bw_mask, interior_area)
    f['n_large_lesions'] = int(n_large)
    f['n_lesions_split'] = int(split_total)
    return f

def extract_mrmf_features(gray, bw_blemish, interior, interior_area):
    """MRMF 20-dimensional vector = base 10 + granulometry (6) + per-lesion (4)."""
    feats = extract_morphological_features(gray, bw_blemish, interior, interior_area)
    feats.update(extract_granulometry_features(gray, interior))
    feats.update(extract_per_lesion_features(bw_blemish, interior_area))
    return feats

def grade_severity(area_ratio, max_lesion_ratio):
    """
    Supplemental: surface quality grading from blemish quantification
    (Assignment Topic 2 scope: 'quantification of blemishes or damage
    relative to the total surface area').

    Grade A: clean surface (< 5% blemish coverage, no dominant lesion)
    Grade B: minor blemishes (5-15% coverage)
    Grade C: significant decay (> 15% coverage, or any single lesion > 10% of fruit)
    """
    if max_lesion_ratio > 10.0:
        return 'Grade C (Significant Decay)'
    if area_ratio < 5.0:
        return 'Grade A (Clean)'
    if area_ratio < 15.0:
        return 'Grade B (Minor Blemishes)'
    return 'Grade C (Significant Decay)'

def analyze_ripeness_by_morphology(image_bgr: np.ndarray, model_path: str = "output/morphology_based/morphology_model.joblib"):
    """
    Herman's Multi-Representation Morphological Fusion (MRMF) inference.

    Returns:
        prediction (str): 'unripe', 'fully_ripe', or 'overripe'
        confidence (float): Classification confidence percentage (0-100%)
        visualized_img (np.ndarray): Image with red blemish overlay + largest-lesion callout, RGB
        metrics (dict): Morphological metrics (defect %, severity grade, per-lesion stats)
        step_images (dict): Intermediate pipeline representations for the UI
    """
    t_start = time.time()

    if not os.path.exists(model_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Morphology model package not found at: {model_path}")

    pkg = _load_model_package(os.path.abspath(model_path))
    model = pkg['model']
    feature_cols = pkg['feature_cols']

    # 1. Masking & Preprocessing
    mask = get_fruit_mask(image_bgr)
    interior = get_interior_mask(mask)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    interior_area = int(np.sum(interior > 0))

    # 2. MRMF Morphology Pipeline
    bw_clean, vis = run_enhanced_blemish_mask(gray, interior)
    feats = extract_mrmf_features(gray, bw_clean, interior, interior_area)

    # 3. Model Prediction
    feat_df = pd.DataFrame([feats])[feature_cols]
    pred_cls = model.predict(feat_df)[0]
    pred_prob = model.predict_proba(feat_df)[0]
    conf = float(np.max(pred_prob) * 100.0)
    latency_ms = (time.time() - t_start) * 1000.0

    # 4. Visualization Overlay (blemish tint + largest-lesion callout)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()
    overlay[bw_clean > 0] = [255, 0, 0]  # Red blemish mask
    blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_clean, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP]
        w, h = stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(blended, (x, y), (x + w, y + h), (255, 255, 0), 3)
        cv2.putText(blended, 'largest lesion', (x, max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    metrics = {
        'blemish_area_ratio': feats['area_ratio'],
        'defect_percentage': round(feats['area_ratio'], 2),
        'severity_grade': grade_severity(feats['area_ratio'], feats['max_lesion_ratio']),
        'needs_review': bool(conf < 70.0),
        'max_lesion_ratio': feats['max_lesion_ratio'],
        'n_lesions_split': feats['n_lesions_split'],
        'n_blemishes': feats['n_blemishes'],
        'mean_darkness': feats['mean_darkness'],
        'skeleton_length': feats['skeleton_length'],
        'latency_ms': latency_ms,
        'features': feats
    }

    bh19 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _ellipse(19))
    bh19 = cv2.normalize(cv2.bitwise_and(bh19, bh19, mask=interior), None, 0, 255, cv2.NORM_MINMAX)

    step_images = {
        '1. Eroded Fruit Mask': interior,
        '2. Beucher Gradient (SE 5x5)': vis['grad'],
        '3. Black-Hat Granulometry (SE 19)': bh19,
        '4. Blemish Mask (Adaptive P92)': bw_clean,
        '5. Blemish Overlay + Lesion Callout': blended,
        '6. Watershed Lesion Split (Yellow Boundaries)': count_split_lesions(
            bw_clean, interior_area, return_vis=True)[2]
    }

    return pred_cls, conf, blended, metrics, step_images
