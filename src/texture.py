import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def analyze_ripeness_by_texture(image: np.ndarray):
    """
    Kai Bin's refactored function: Texture Analysis
    Uses Gray-Level Co-occurrence Matrix (GLCM) features like contrast and energy to classify ripeness.
    
    Returns:
        prediction (str): 'Unripe', 'Partially Ripe', or 'Fully Ripe'
        confidence (float): Confidence score
        visualized_img (np.ndarray): Texture map visualization
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to speed up GLCM calculation
    gray_resized = cv2.resize(gray, (256, 256))
    
    # Calculate GLCM
    glcm = graycomatrix(gray_resized, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Normalize features to some expected range for classification
    # Typically, Unripe = lower contrast, higher energy/homogeneity (smoother skin)
    # Partially Ripe = medium contrast, medium energy
    # Fully Ripe = higher contrast, lower energy (wrinkles, spots)
    
    # Using simple thresholds based on typical mango texture profiles
    # (These thresholds are illustrative and can be fine-tuned)
    if contrast < 100:
        prediction = "Unripe"
        confidence = min(1.0, 1.0 - (contrast / 100) * 0.3)
    elif contrast > 250:
        prediction = "Fully Ripe"
        confidence = min(1.0, (contrast / 400))
    else:
        prediction = "Partially Ripe"
        confidence = 0.8
        
    # Generate visualization: Sobel filter to show texture intensity/gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    
    visualized = cv2.applyColorMap(sobel, cv2.COLORMAP_JET)
    
    return prediction, float(confidence), visualized
