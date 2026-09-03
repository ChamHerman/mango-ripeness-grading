# Mango Ripeness Grading & Inspection System

An automated classical computer vision and image processing suite for non-destructive grading, maturity classification, and real-time quality control of mango fruits (*Mangifera indica*). Developed as a collaborative multi-algorithmic prototype for **BMDS2133 Image Processing**.

---

## Key Features & Highlights

- **4 Complementary Classical CV Engines**: Integrates morphology, chrominance color spaces, statistical texture, and contour morphometry without relying on black-box deep learning.
- **Hybrid Ensemble Majority Consensus**: Plurality voting with cumulative confidence tie-breaking ensures robust decisions and complete immunity against isolated rogue outlier predictions.
- **Real-Time Multi-Fruit Detection (30+ FPS)**: Simultaneous localization, tracking, and ripeness classification via OpenCV DirectShow hardware camera, browser WebRTC, or video upload.
- **High-Throughput Conveyor Simulation**: Batch assessment processing directory streams, multi-image uploads, or uploaded `.zip` archives with interactive telemetry.
- **Industrial Automated PDF Reports**: One-click downloadable quality inspection reports formatted with clean, standard terminology (`Fully Ripe`, `Overripe`, `Unripe`) via ReportLab.
- **Hardware Acceleration**: Automatic GPU acceleration via OpenCV OpenCL / CUDA with seamless multi-threaded CPU SIMD fallback.

---

## Core Modules & Team Contributions

The system integrates four distinct classical computer vision and statistical feature extraction pipelines:

| Module | Developer | Core Formulation | Test Accuracy | Latency | Key Physical Ripeness Cues |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Morphological Blemish Analysis** | **Cham Herman** *(Lead / Fusion)* | Multi-Scale Beucher Gradient & Black-Hat Residual Fusion (MRMF) + Random Forest | **98.61%** | 32.5 ms | Anthracnose lesions, surface blemishes, defect severity grading (`Grade A/B/C`) |
| **Color-Space Analysis** | **Lum Siew Feng** *(Color Engineer)* | Multi-Space Chrominance Extraction (RGB, HSV, LAB, YCbCr, HLS) + RBF SVM | **100.00%** *(LAB)* | 12.5 ms | Chlorophyll degradation & carotenoid accumulation |
| **Texture & Surface Analysis** | **Wong Kai Bin** *(Texture Lead)* | Rotation-Invariant GLCM (4 angles) + Uniform Local Binary Patterns (LBP) + RBF SVM | **92.36%** | 18.3 ms | Peel micro-roughness, lenticel speckle, and textural entropy |
| **Edge & Shape Deformity** | **Yeow Wei Kang** *(Geometry Lead)* | Scharr Edge Density Gradient + Multi-Parametric Contour Morphometry + ExtraTrees | **91.67%** | 25.0 ms | Fruit softening, contour shoulder shrinkage, circularity & aspect ratio |

---

## System Architecture & Application Pages

The interactive Streamlit application provides four dedicated operational dashboards:

### 1. Single Image Diagnostic Playground
- **Side-by-Side Diagnostic Views**: Run single or all four algorithms concurrently with individual model cards.
- **Shared Upstream Preprocessing Pipeline**: Interactive step-by-step inspector tracking transformations from P1 (Letterbox) to P6 (Background Masking).
- **Intermediate Pipeline Diagnostics**: Detailed 7-step intermediate visualization for each individual pipeline, including an interactive 5-color-space switcher (`RGB`, `HSV`, `LAB`, `YCbCr`, `HLS`).
- **Hybrid Ensemble Consensus Verdict**: Real-time plurality decision banner displaying consensus class, average confidence score, and total cumulative latency.
- **PDF Export**: Instant single-image quality inspection report generation.

### 2. Bulk Batch Assessment (Conveyor Stream)
- **Multi-Source Stream Ingestion**:
  - Direct repository directories (`cleaned_data/test/`, `cleaned_data/train/`, `data/`).
  - Drag-and-drop `.zip` dataset archive upload with automatic extraction and non-image filtering.
  - Multi-file image picker for arbitrary custom batches.
- **Configurable Preprocessing**: Choose between high-speed Morphological Masking (~3–5 ms) or K-Means Color Clustering (~150–300 ms).
- **Hybrid Ensemble Majority Consensus Decision Fusion**:
  - Automatically tallies votes across all active modules.
  - Plurality vote governs the verdict; tied votes are broken using cumulative confidence scores.
  - Single-model direct passthrough when only one engine is active.
  - Prevents an individual overconfident model from corrupting batch quality grades.
- **Visual Analytics**: High-contrast Batch Maturity Distribution chart (white background for full visibility in dark and light modes) and summary KPI counters (`Total Inspected`, `Fully Ripe (Pass)`, `Unripe (Hold)`, `Overripe (Reject)`, `Avg Confidence`).
- **Itemized Telemetry & Reports**: Interactive data table with per-image diagnostics, winning attribution (`Ensemble Majority` vs `Unanimous`), and downloadable multi-sample PDF inspection report.

### 3. Real-Time Multi-Mango Detection & Ripeness Counting
- **Multi-Stream Video Ingestion**:
  - *Direct Hardware Camera* (OpenCV DirectShow as default — ultra-low latency, zero browser overhead).
  - *Browser WebRTC Stream* (via `streamlit-webrtc`).
  - *Pre-recorded Video Upload* (`.mp4`, `.avi`, `.mov`).
- **Real-Time Multi-Instance Localization**: Connected component and contour analysis for simultaneous multi-mango localization, counting, and individual ROI extraction.
- **Live Classification & Consensus**: Live classification across any combination of the 4 pipelines with hybrid consensus voting.
- **Augmented Reality HUD**: In-frame bounding boxes, corner tech accents, and floating ripeness verdict badges overlaid directly onto each tracked fruit.
- **Live Stream Telemetry**: Rolling real-time FPS counter, per-frame latency gauge, active mango count, distribution tally, and exportable CSV telemetry logs.

### 4. System Analytics & Comparative Benchmark
- **Mode A Table 2.1**: Comprehensive comparative benchmark displaying feature vector dimensions, classifier architectures, test accuracies, and inference latencies.
- **Environmental Robustness & Invariance Matrix**: Theoretical invariance properties (Illumination and Scale Invariance) and biological cues targeted by each technique.
- **Verified Performance Visualizations**: Side-by-side bar charts comparing test accuracy against the $\ge 85\%$ threshold and latency against the $200\text{ ms}$ real-time budget.
- **5-Color Space Benchmark Breakdown**: Detailed accuracy and ranking comparison across `RGB`, `HSV`, `LAB`, `YCbCr`, and `HLS`.
- **SMART Objectives Verification**: Formal validation proving all project objectives were achieved and exceeded.

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
├── cleaned_data/                  # Cleaned and verified train/test splits
│   ├── train/                     # Training split (571 images across 3 classes)
│   └── test/                      # Test split (144 images across 3 classes)
│
├── data/                          # Raw reference dataset (unripe, partially_ripe, fully_ripe)
│
├── docs/                          # Architecture Decision Records (ADRs) & specifications
│
├── notebooks/                     # Research and algorithmic exploration notebooks
│   ├── color_space_sf.ipynb       # Siew Feng: Multi-Color Space & SVM exploration
│   ├── edge_detection_wk.ipynb    # Wei Kang: Scharr Edge Density & Contour Morphometry
│   ├── morphological_analysis_hm.ipynb # Herman: MRMF Morphological Blemish Analysis
│   └── texture_analysis_kb.ipynb  # Kai Bin: GLCM & LBP Texture Analysis
│
├── src/                           # Production source modules
│   ├── __init__.py
│   ├── benchmark.py               # Dynamic benchmark metric caching & evaluation
│   ├── color_space.py             # Color space chrominance extractors & SVM model
│   ├── dataset_cleaning.py        # Dataset validation, verification & deduplication
│   ├── geometry.py                # Scharr edge density & contour geometric features
│   ├── hardware.py                # GPU (CUDA/OpenCL) auto-detection & dispatcher
│   ├── morphology.py              # Multi-Scale Beucher & Black-Hat residual fusion (MRMF)
│   ├── preprocessing.py           # Letterboxing, denoising, CLAHE, segmentation
│   ├── realtime_detection.py      # Real-time multi-mango contour tracker & live HUD
│   ├── reports.py                 # ReportLab PDF quality inspection report generator
│   ├── texture.py                 # Rotation-invariant GLCM & LBP feature extraction
│   └── video.py                   # WebRTC callbacks, HUD overlay & telemetry logging
│
├── app.py                         # Streamlit multi-page dashboard application
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Installation & Usage

### 1. Environment Setup

Clone the repository and create an isolated Python virtual environment (`.venv`):

```bash
# Clone the repository
git clone https://github.com/ChamHerman/mango-ripeness-grading.git
cd mango-ripeness-grading

# Create virtual environment
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard

Always run Streamlit inside the project virtual environment (`.venv`) to ensure consistent configurations and dependencies:

```powershell
# Using active virtual environment:
streamlit run app.py

# Or run directly via the virtual environment executable:
.venv\Scripts\python.exe -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Evaluation Benchmark Summary

| Objective | Target Criterion | Measured Benchmark Status | Fulfillment |
| :--- | :--- | :--- | :---: |
| **Multi-Algorithm Suite** | Implement 4 distinct classical computer vision algorithms | 4 Modules Integrated (Morphology, Color, Texture, Geometry) | **Achieved** (100% Finalized) |
| **Classification Accuracy** | Minimum $\ge 85\%$ accuracy across all modules | 91.67% (Geometry) to 100.00% (LAB Color) | **Target Exceeded** |
| **Operational Latency** | Per-image processing budget $< 200\text{ ms}$ | 12.45 ms (Color) to 32.48 ms (Morphology) | **Target Exceeded** |
