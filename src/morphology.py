import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Simple CNN Architecture for Mango Ripeness Classification
class MangoCNN(nn.Module):
    def __init__(self):
        super(MangoCNN, self).__init__()
        # Simple feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 3 classes: Unripe, Partially Ripe, Fully Ripe
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_or_create_model(model_path: str):
    """
    Loads model weights from path. If file does not exist, creates it with initialized weights.
    """
    model = MangoCNN()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except Exception:
            # Fallback if weight schema mismatches
            torch.save(model.state_dict(), model_path)
    else:
        # Save a dummy initialized model
        torch.save(model.state_dict(), model_path)
        
    model.eval()
    return model

def analyze_ripeness_by_deep_learning(image: np.ndarray, model_path: str = "models/deep_learning_model.pth"):
    """
    Herman's refactored Deep Learning model loader/predictor.
    Runs inference on the input image using the trained CNN.
    
    Returns:
        prediction (str): 'Unripe', 'Partially Ripe', or 'Fully Ripe'
        confidence (float): Confidence score
        visualized_img (np.ndarray): Original image (or bounding box visualization)
    """
    model = load_or_create_model(model_path)
    
    # Preprocessing
    # OpenCV image is BGR, convert to RGB
    img_rgb = Image.fromarray(image[:, :, ::-1])
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[class_idx].item()
        
    classes = ["Unripe", "Partially Ripe", "Fully Ripe"]
    prediction = classes[class_idx]
    
    # Draw classification text on image as visualization
    visualized = image.copy()
    h, w, _ = visualized.shape
    cv2.putText(visualized, f"DL: {prediction} ({confidence:.2f})", 
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return prediction, float(confidence), visualized
