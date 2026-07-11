import cv2
import numpy as np

def analyze_ripeness_by_color(image: np.ndarray):
    """
    Siew Feng's refactored function: Color Thresholding / Space Analysis
    Analyzes mango ripeness based on color distribution (HSV space).
    
    Returns:
        prediction (str): 'Unripe', 'Partially Ripe', or 'Fully Ripe'
        confidence (float): Confidence score
        visualized_img (np.ndarray): Color-segmented visualization image
    """
    # Ensure image is in BGR format (OpenCV default)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Green (Unripe)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Yellow/Orange/Red (Fully Ripe)
    lower_yellow = np.array([10, 40, 40])
    upper_yellow = np.array([34, 255, 255])
    
    # Create masks
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = np.sum(mask_green > 0)
    yellow_pixels = np.sum(mask_yellow > 0)
    
    # Percentages relative to total (or mask totals)
    sum_detected = green_pixels + yellow_pixels
    if sum_detected == 0:
        return "Unknown", 0.0, image
        
    green_pct = green_pixels / sum_detected
    yellow_pct = yellow_pixels / sum_detected
    
    # Classification logic
    if green_pct > 0.65:
        prediction = "Unripe"
        confidence = green_pct
    elif yellow_pct > 0.65:
        prediction = "Fully Ripe"
        confidence = yellow_pct
    else:
        prediction = "Partially Ripe"
        confidence = 1.0 - abs(green_pct - yellow_pct)
        
    # Visualization: highlight green and yellow masks
    visualized = image.copy()
    visualized[mask_green > 0] = [0, 255, 0]    # Green highlight
    visualized[mask_yellow > 0] = [0, 255, 255] # Yellow highlight
    
    return prediction, float(confidence), visualized
