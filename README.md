# 🥭 Mango Ripeness Grading & Analytics Dashboard

This repository contains the image processing and grading suite designed to classify mango ripeness using four distinct computational approaches:
1. **Color Space Analysis (HSV/LAB)** — *Developed by Siew Feng*
2. **GLCM Texture Feature Analysis** — *Developed by Kai Bin*
3. **Canny/Sobel Geometry & Edges** — *Developed by Wei Kang*
4. **Deep Learning (PyTorch CNN)** — *Developed by Cham Herman*

---

## 📁 Repository Structure

```
mango-ripeness-grading/
│
├── data/                          # Dataset organized by ripeness stage
│   ├── unripe/                    # Unripe mango images (.gitkeep placeholder)
│   ├── partially_ripe/            # Partially ripe mango images (.gitkeep placeholder)
│   └── fully_ripe/                # Fully ripe mango images (.gitkeep placeholder)
│
├── models/                        # Saved weight assets for deep learning models
│   └── deep_learning_model.pth    # PyTorch serialized weights
│
├── notebooks/                     # Exploratory jupyter playgrounds
│   ├── color_thresholding_sf.ipynb# Siew Feng's playground
│   ├── texture_analysis_kb.ipynb  # Kai Bin's playground
│   ├── edge_detection_wk.ipynb    # Wei Kang's playground
│   └── deep_learning_hm.ipynb     # Herman's playground
│
├── src/                           # Productionized python modules
│   ├── __init__.py
│   ├── color_space.py             # Color space operations
│   ├── texture.py                 # GLCM extraction
│   ├── geometry.py                # Canny edges & contours
│   ├── deep_learning.py           # PyTorch CNN loader/predictor
│   └── reports.py                 # PDF generation utility
│
├── app.py                         # Streamlit GUI application
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit GUI**:
   ```bash
   streamlit run app.py
   ```
