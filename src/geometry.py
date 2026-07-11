import cv2
import numpy as np

def analyze_ripeness_by_geometry(image: np.ndarray):
    """
    Wei Kang's refactored function: Edge detection and Geometry
    Uses Canny edge density and contour analysis (circularity/aspect ratio) to analyze ripeness.
    
    Returns:
        prediction (str): 'Unripe', 'Partially Ripe', or 'Fully Ripe'
        confidence (float): Confidence score
        visualized_img (np.ndarray): Edge detection visualization
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Unripe: smooth contour, fewer internal edges (low density)
    # Fully Ripe: more internal edges/wrinkles/blemish contours (high density)
    if edge_density < 0.02:
        prediction = "Unripe"
        confidence = min(1.0, 1.0 - (edge_density / 0.02) * 0.4)
    elif edge_density > 0.05:
        prediction = "Fully Ripe"
        confidence = min(1.0, edge_density * 15.0)
    else:
        prediction = "Partially Ripe"
        confidence = 0.75
        
    # Visualization: Overlay edges on BGR image
    visualized = image.copy()
    visualized[edges > 0] = [0, 0, 255] # Draw edges in red
    
    # Draw contour bounding box if contours found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(visualized, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return prediction, float(confidence), visualized
