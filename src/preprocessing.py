import cv2
import numpy as np


# --------------------------------------------------
# Resize Image
# --------------------------------------------------
def resize_image(image, size=(640, 640)):
    if image is None:
        raise ValueError("Input image is None. Please verify that cv2.imread() successfully loaded the image file.")
    h, w = image.shape[:2]
    target_w, target_h = size

    # calculate scaling ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize while keeping ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # create blank canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # calculate padding
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # place image
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# --------------------------------------------------
# Median Filtering
# --------------------------------------------------
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# --------------------------------------------------
# CLAHE Contrast Enhancement
# --------------------------------------------------
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ============================================================
# Generate Binary Mask via Color Clustering (K-Means + HSV + Convex Hull)
# ============================================================
def generate_mango_mask(image, k_clusters=6, dark_spot_dist_threshold=30):
    """
    Generate binary mask using K-Means color clustering, HSV color classification,
    and Convex Hull to isolate the mango fruit.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Failed to load image from path: {image}")
    else:
        img = image.copy()

    # Step 1: Detect dominant colors using K-Means Clustering
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    data = blurred.reshape((-1, 3)).astype(np.float32)

    # Define criteria and perform K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels_img = labels.reshape(img.shape[:2])
    centers_bgr = np.uint8(centers)
    centers_hsv = cv2.cvtColor(np.expand_dims(centers_bgr, axis=0), cv2.COLOR_BGR2HSV)[0]

    mango_color_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    dark_defect_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Step 2: Spot the mango colors (Green, Yellow, Dark Red) vs. Dark spots
    for i, (h, s, v) in enumerate(centers_hsv):
        cluster_pixels = (labels_img == i).astype(np.uint8) * 255

        is_green = (36 <= h <= 85) and (s >= 40) and (v >= 40)
        is_yellow = (15 <= h <= 35) and (s >= 65) and (v >= 50)
        is_red = ((0 <= h <= 10) or (170 <= h <= 180)) and (s >= 80) and (v >= 40)
        is_dark = (v < 55)

        if is_green or is_yellow or is_red:
            mango_color_mask = cv2.bitwise_or(mango_color_mask, cluster_pixels)
        elif is_dark:
            dark_defect_mask = cv2.bitwise_or(dark_defect_mask, cluster_pixels)

    # Step 3: Find the primary mango body contour
    contours_body, _ = cv2.findContours(mango_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_body:
        h_img, w_img = img.shape[:2]
        return np.ones((h_img, w_img), dtype=np.uint8)

    main_mango_contour = max(contours_body, key=cv2.contourArea)

    # Step 4: Include nearby dark spots touching or near the mango edge
    dark_contours, _ = cv2.findContours(dark_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    included_contours = [main_mango_contour]
    for d_cnt in dark_contours:
        if cv2.contourArea(d_cnt) < 15:
            continue

        dist = -1
        for pt in d_cnt:
            d = cv2.pointPolygonTest(main_mango_contour, (float(pt[0][0]), float(pt[0][1])), True)
            if dist == -1 or abs(d) < dist:
                dist = abs(d)

        if dist <= dark_spot_dist_threshold:
            included_contours.append(d_cnt)

    # Step 5: Merge contours and wrap with Convex Hull
    all_mango_points = np.vstack(included_contours)
    final_hull = cv2.convexHull(all_mango_points)

    h_img, w_img = img.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(mask, [final_hull], -1, 1, thickness=cv2.FILLED)

    return mask


# ============================================================
# Morphological Cleaning
# ============================================================
def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ============================================================
# Background Removal
# ============================================================
def remove_background(image, k_clusters=6, dark_spot_dist_threshold=30):
    """
    Remove background using color clustering, morphological cleaning, and masking.
    Returns: (masked_image, cleaned_mask)
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Failed to load image from path: {image}")
    else:
        img = image.copy()

    mask = generate_mango_mask(img, k_clusters=k_clusters, dark_spot_dist_threshold=dark_spot_dist_threshold)
    cleaned = clean_mask(mask)
    masked_image = cv2.bitwise_and(img, img, mask=cleaned)
    return masked_image, cleaned

# --------------------------------------------------
# Complete Preprocessing Pipeline
# --------------------------------------------------
def preprocess_image(image, k_clusters=6, dark_spot_dist_threshold=30, **kwargs):
    image = resize_image(image)
    image = remove_noise(image)
    image = enhance_contrast(image)
    image, mask = remove_background(image, k_clusters=k_clusters, dark_spot_dist_threshold=dark_spot_dist_threshold)
    return image
