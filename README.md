# Mango Ripeness Grading & Inspection System

An automated classical computer vision and image processing suite for non-destructive grading and maturity classification of mango fruits. Developed as a collaborative multi-algorithmic prototype for **BMDS2133 Image Processing**.

---

## Core Modules & Team Contributions

The system integrates four distinct classical computer vision and statistical feature extraction pipelines:

1. **Morphological Blemish Analysis (Cham Herman)**
   - **Formulation**: Multi-Scale Beucher Morphological Gradient & Dual Black-Hat Residual Fusion (MRMF).
   - **Function**: Extracts surface lesions, necrotic blemish ratios, and defect severity ratings (`Grade A`, `Grade B`, `Grade C`).
   - **Performance**: 98.61% Test Accuracy | 32.5 ms Latency.

2. **Color-Space Chrominance Analysis (Lum Siew Feng)**
   - **Formulation**: Multi-Color Space Chrominance Extraction (RGB, HSV, LAB, YCbCr, HLS) evaluated with Support Vector Classification (SVM).
   - **Function**: Classifies ripeness transitions based on yellow-to-green chrominance shifts and color clustering.
   - **Performance**: 100.00% Best Test Accuracy (LAB) | 12.5 ms Latency.

3. **Texture & Surface GLCM Analysis (Wong Kai Bin)**
   - **Formulation**: Rotation-Invariant Gray-Level Co-occurrence Matrix (GLCM across 4 angles) + Uniform Local Binary Patterns (LBP) + Surface Roughness.
   - **Function**: Captures micro-textural changes, peel smoothness, and surface entropy.
   - **Performance**: 92.36% Test Accuracy | 18.3 ms Latency.

4. **Edge & Shape Deformity Detection (Yeow Wei Kang)**
   - **Formulation**: Scharr Gradient Edge Density + Multi-Parametric Contour Geometry (Aspect Ratio, Extent, Solidity, Perimeter/Area).
   - **Function**: Detects morphological deformity, contour boundaries, and structural irregularities.
   - **Performance**: 91.67% Test Accuracy | 25.0 ms Latency.

---

## System Architecture & Application Pages

The Streamlit web application is structured into four functional pages:

### 1. Single Image Diagnostic Playground
- Multi-algorithm side-by-side execution with independent toggle controls.
- Dedicated shared upstream preprocessing inspector (P1 Letterbox to P6 Background Masking).
- Interactive step-by-step intermediate pipeline expanders for each algorithm.
- Consensus decision voting and defect confidence scoring.

### 2. Bulk Batch Assessment (Conveyor Stream)
- High-throughput conveyor simulation processing multiple images concurrently.
- Flexible input sources: direct local repository directories (`data/`, `cleaned_data/test/`, `cleaned_data/train/`) or bulk `.zip` archive upload.
- Batch defect distribution charts and summary KPI telemetry.
- Automated multi-page PDF quality inspection report generation via ReportLab.

### 3. Live Camera Inspection (Real-Time Stream)
- Low-latency webcam stream ingestion powered by Streamlit-WebRTC and PyAV.
- Configurable upstream preprocessing selection:
  - *Standard K-Means Color Clustering & Convex Hull (Default)*
  - *Background-Agnostic Morphological Fruit Segmentation*
- Dynamic hot-switching across all 4 classification pipelines during streaming.
- Two-line HUD banner with text measurement to prevent label collision.
- Side-by-side stream layout with manual stream controls (`Start Live Stream` / `Stop Live Stream`).
- Real-time diagnostic telemetry panel with 18-metric CSV log export (`Download Telemetry Log`).

### 4. System Analytics & Comparative Benchmark
- Mode A Table 2.1 comparative benchmark across all team modules.
- Detailed accuracy breakdown across all 5 evaluated color spaces (RGB, HSV, LAB, YCbCr, HLS).
- Verified performance charts comparing accuracy and latency against the 85% accuracy and 33.3 ms real-time streaming thresholds.
- Hardware compute device status and SMART objective verification matrix.

---

## Hardware Compute & GPU Acceleration

The system includes an automatic compute device dispatcher (`src/hardware.py`):
- **GPU Acceleration**: Automatically detects NVIDIA GPUs (via OpenCV OpenCL / CUDA) and dispatches matrix operations, morphological filtering, color conversions, and edge convolutions to the GPU.
- **CPU SIMD Fallback**: Seamlessly falls back to multi-threaded CPU SIMD execution (OpenMP / AVX2) if no dedicated GPU is available.

---

## Repository Structure

```
mango-ripeness-grading/
│
├── cleaned_data/                  # Validated and cleaned train/test splits
│   ├── train/                     # Training split organized by class
│   └── test/                      # Test split organized by class
│
├── data/                          # Raw dataset (unripe, partially_ripe, fully_ripe)
│
├── docs/                          # Architecture Decision Records (ADRs) & specs
│
├── notebooks/                     # Exploratory research notebooks
│   ├── color_space_sf.ipynb       # Siew Feng: Color Space & SVM exploration
│   ├── edge_detection_wk.ipynb    # Wei Kang: Scharr Edge & Contour Geometry
│   ├── morphological_analysis_hm.ipynb # Herman: MRMF Morphological Blemish
│   └── texture_analysis_kb.ipynb  # Kai Bin: GLCM & LBP Texture Analysis
│
├── src/                           # Production source modules
│   ├── __init__.py
│   ├── benchmark.py               # Benchmark metric caching & Table 2.1 compilation
│   ├── color_space.py             # Color space chrominance extractors & SVM model
│   ├── dataset_cleaning.py        # Dataset validation & deduplication utilities
│   ├── geometry.py                # Scharr edge density & contour geometric features
│   ├── hardware.py                # GPU (CUDA/OpenCL) auto-detection & dispatcher
│   ├── morphology.py              # Multi-Scale Beucher & Black-Hat residual fusion
│   ├── preprocessing.py           # Letterboxing, denoising, CLAHE, segmentation
│   ├── reports.py                 # ReportLab PDF quality report generator
│   ├── texture.py                 # Rotation-invariant GLCM & LBP feature extraction
│   └── video.py                   # WebRTC callbacks, HUD overlay & telemetry logging
│
├── app.py                         # Streamlit multi-page dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## Installation & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ChamHerman/mango-ripeness-grading.git
cd mango-ripeness-grading

# Create and activate a Python virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
