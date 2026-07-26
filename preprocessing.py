import cv2
import numpy as np

# Try importing rembg with fallback handling
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg module not found. Falling back to GrabCut.")

# --------------------------------------------------
# Resize Image
# --------------------------------------------------
def resize_image(image, size=(640, 640)):
    h, w = image.shape[:2]

    target_w, target_h = size


    # calculate scaling ratio
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)


    # resize while keeping ratio
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


    # create blank canvas
    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8
    )


    # calculate padding
    x_offset = (target_w - new_w)//2
    y_offset = (target_h - new_h)//2


    # place image
    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized


    return canvas
    # return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


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

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l,a,b))

    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ============================================================
# Background Removal using GrabCut
# ============================================================
def remove_background(image):

    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    height, width = image.shape[:2]

    rect = (
        10,
        10,
        width-20,
        height-20
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    mask = np.where(
        (mask==2)|(mask==0),
        0,
        1
    ).astype("uint8")

    return image * mask[:, :, np.newaxis], mask

# ============================================================
# Background Removal using rembg (AI model)
# ============================================================
def remove_background_rembg(image):
    if not REMBG_AVAILABLE:
        return remove_background(image)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_rgba = remove(rgb_image)
    output_rgb = output_rgba[:, :, :3]
    mask = (output_rgba[:, :, 3] > 0).astype(np.uint8)
    bgr_output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    return cv2.bitwise_and(bgr_output, bgr_output, mask=mask), mask

# ============================================================
# Morphological Cleaning
# ============================================================
def clean_mask(mask):

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

# ============================================================
# Apply Clean Mask
# ============================================================
def apply_mask(image, mask):

    return cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

# --------------------------------------------------
# Complete Preprocessing Pipeline
# --------------------------------------------------
def preprocess_image(image, use_rembg=True):

    image = resize_image(image)
    image = remove_noise(image)
    image = enhance_contrast(image)
    if use_rembg and REMBG_AVAILABLE:
        image, mask = remove_background_rembg(image)
    else:
        image, mask = remove_background(image)
    mask = clean_mask(mask)
    image = apply_mask(image, mask)
    return image
