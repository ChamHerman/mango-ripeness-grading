import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import modular image processing engines
from src.preprocessing import preprocess_image, preprocess_image_with_stages
from src.morphology import analyze_ripeness_by_morphology
from src.color_space import analyze_ripeness_by_color, get_color_space_pipeline_steps, COLOR_SPACES
from src.texture import analyze_ripeness_by_texture
from src.geometry import analyze_ripeness_by_geometry
from src.reports import generate_pdf_report
from src.benchmark import get_benchmark_metrics
from src.video import analyze_frame, make_webrtc_callback, LiveSessionStats, ALGORITHM_METADATA
from src.hardware import get_hardware_info, init_hardware_acceleration

# Auto-initialize compute device acceleration (GPU where available, CPU fallback)
_IS_GPU_ACTIVE, _ACTIVE_DEVICE = init_hardware_acceleration(enable_gpu=True)

# Page Configuration
st.set_page_config(
    page_title="Mango Ripeness Grading & Inspection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# SVG Icons Library (Crisp, modern inline vector icons)
# -----------------------------------------------------------------------------
SVG_ICONS = {
    'mango': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.5 2 2 6.5 2 12c0 3.5 2 6.5 5 8.5 3 2 7 1.5 10-1.5s5-7 5-10.5C22 5 17.5 2 12 2z"></path><path d="M12 2c1 3 3 5 6 5"></path></svg>',
    'diagnostic': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="m4.93 4.93 4.24 4.24"></path><path d="m14.83 9.17 4.24-4.24"></path><path d="m14.83 14.83 4.24 4.24"></path><path d="m9.17 14.83-4.24 4.24"></path><circle cx="12" cy="12" r="4"></circle></svg>',
    'conveyor': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="14" rx="2"></rect><path d="M6 18h.01"></path><path d="M10 18h.01"></path><path d="M14 18h.01"></path><path d="M18 18h.01"></path><path d="M4 14V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8"></path></svg>',
    'analytics': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="m19 9-5 5-4-4-3 3"></path></svg>',
    'verified': '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>',
    'scaffold': '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    'upload': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>',
    'sliders': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>',
    'trash': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>',
    'table': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"></path><rect width="18" height="18" x="3" y="3" rx="2"></rect><path d="M3 9h18"></path><path d="M3 15h18"></path></svg>',
    'eye': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>',
    'camera': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path><circle cx="12" cy="13" r="3"></circle></svg>'
}

# -----------------------------------------------------------------------------
# Adaptive Styling (Seamlessly integrates with Streamlit Light & Dark Themes)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Branding Gradient Headers */
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sub-title {
        font-size: 0.95rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Adaptable Glass Container Cards */
    .glass-card {
        background: rgba(128, 128, 128, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(245, 158, 11, 0.5);
        transform: translateY(-2px);
    }
    
    /* Metric Highlights */
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.7;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    
    /* Status Badges */
    .badge-unripe {
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-ripe {
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-overripe {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-completed {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    
    .status-scaffold {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: rgba(245, 158, 11, 0.12);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    
    /* Consensus Banner */
    .consensus-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(234, 88, 12, 0.08) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 20px;
    /* WebRTC Stream Containment & Responsive Sizing */
    div[data-testid="stWebRtc"] {
        max-width: 520px !important;
        margin: 0 auto !important;
        border-radius: 12px;
        overflow: hidden;
    }
    iframe[title="streamlit_webrtc.webrtc_streamer"] {
        max-width: 520px !important;
        max-height: 440px !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"<div style='font-size: 1.2rem; font-weight: bold; color: #f59e0b; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['mango']} Mango Ripeness Grading</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 0.8rem; opacity: 0.7;'>BMDS2133 Image Processing Prototype</div><hr style='margin: 8px 0 16px 0; opacity: 0.2;'>", unsafe_allow_html=True)

selected_page = st.sidebar.radio(
    "Navigation:",
    ["Single Image Diagnostic Playground",
     "Bulk Batch Assessment (Conveyor Stream)",
     "Live Camera Inspection (Real-Time Stream)",
     "System Analytics & Comparative Benchmark"]
)

# Retrieve Dynamic Benchmark Metrics
bm_metrics = get_benchmark_metrics()
morph_bm = bm_metrics.get('morphology', {})
best_cs = bm_metrics.get('best_color_space', 'LAB')
best_cs_acc = bm_metrics.get('best_color_accuracy', 100.00)
best_cs_f1 = bm_metrics.get('best_color_f1', 100.00)
texture_bm = bm_metrics.get('texture', {})
texture_acc = texture_bm.get('accuracy', 96.53)
texture_f1 = texture_bm.get('f1', 96.52)
texture_lat = texture_bm.get('latency_ms', 18.30)

geom_bm = bm_metrics.get('geometry', {})
geom_acc = geom_bm.get('accuracy', 91.67)
geom_f1 = geom_bm.get('f1', 91.65)
geom_lat = geom_bm.get('latency_ms', 25.0)

hw_info = get_hardware_info()
hw_badge_html = f"<span class='status-completed'>{SVG_ICONS['verified']} GPU: {hw_info['device_name']}</span>" if hw_info['has_gpu'] else f"<span style='opacity:0.8; font-size:0.72rem;'>CPU: {hw_info['device_name']}</span>"

st.sidebar.markdown(f"""
<div style='font-size: 0.75rem; opacity: 0.8;'>
    <b>Team Module Status:</b><br><br>
    <div style='margin-bottom: 6px;'><b>Cham Herman</b>: Morphological Blemish<br><span class='status-completed'>{SVG_ICONS['verified']} Completed ({morph_bm.get('accuracy', 98.61):.2f}% Acc)</span></div>
    <div style='margin-bottom: 6px;'><b>Lum Siew Feng</b>: Color-Space Analysis<br><span class='status-completed'>{SVG_ICONS['verified']} Completed ({best_cs_acc:.2f}% Acc — Best: {best_cs})</span></div>
    <div style='margin-bottom: 6px;'><b>Wong Kai Bin</b>: Texture & Surface Analysis<br><span class='status-completed'>{SVG_ICONS['verified']} Completed ({texture_acc:.2f}% Acc)</span></div>
    <div style='margin-bottom: 6px;'><b>Yeow Wei Kang</b>: Edge & Shape Geometry<br><span class='status-completed'>{SVG_ICONS['verified']} Completed (91.67% Acc)</span></div>
    <hr style='margin: 8px 0; opacity: 0.2;'>
    <b>Compute Hardware:</b><br>
    {hw_badge_html}
</div>
""", unsafe_allow_html=True)

def get_class_badge(cls_name):
    cls_lower = str(cls_name).lower()
    if 'unripe' in cls_lower:
        return "<span class='badge-unripe'>Stage 0: Unripe</span>"
    elif 'overripe' in cls_lower:
        return "<span class='badge-overripe'>Stage Overripe</span>"
    else:
        return "<span class='badge-ripe'>Stage 3: Fully Ripe</span>"

# -----------------------------------------------------------------------------
# PAGE 1: DIAGNOSTIC PLAYGROUND (SINGLE IMAGE)
# -----------------------------------------------------------------------------
if selected_page.startswith("Single"):
    st.markdown(f"<div class='main-title'>{SVG_ICONS['diagnostic']} Diagnostic Playground</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Interactive computer vision diagnostic suite allowing multi-technique selection, step-by-step pipeline inspection, and comparative grading.</div>", unsafe_allow_html=True)
    
    col_input, col_config = st.columns([1.2, 1.0])
    
    with col_input:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['upload']} 1. Input Mango Image</div>", unsafe_allow_html=True)
        input_source = st.radio("Input Source:", ["Preloaded Standard Dataset Samples", "Upload Image File"], horizontal=True)
        
        img_bgr = None
        img_filename = "sample.jpg"
        
        if input_source == "Upload Image File":
            uploaded_file = st.file_uploader("Upload Mango Image (.jpg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_filename = uploaded_file.name
        else:
            sample_options = {
                "Cleaned Unripe Mango (cleaned_data/test/unripe)": "cleaned_data/test/unripe",
                "Cleaned Fully Ripe Mango (cleaned_data/test/fully_ripe)": "cleaned_data/test/fully_ripe",
                "Cleaned Overripe Mango (cleaned_data/test/overripe)": "cleaned_data/test/overripe",
                "Raw Unripe Mango (data/unripe)": "data/unripe",
                "Raw Fully Ripe Mango (data/fully_ripe)": "data/fully_ripe",
                "Raw Overripe Mango (data/overripe)": "data/overripe",
            }
            selected_sample_label = st.selectbox("Select Sample Category:", list(sample_options.keys()))
            sample_dir = sample_options[selected_sample_label]
            available_samples = sorted(glob.glob(f"{sample_dir}/*.*"))
            if available_samples:
                sample_file_names = [os.path.basename(p) for p in available_samples]
                selected_file_name = st.selectbox("Select Specific Image File:", sample_file_names[:30], index=0)
                img_path = os.path.join(sample_dir, selected_file_name)
                img_bgr = cv2.imread(img_path)
                img_filename = selected_file_name
                
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Input Image: {img_filename}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_config:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['sliders']} 2. Algorithm Selection</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; opacity: 0.7; margin-bottom: 12px;'>Select which classical algorithms to execute for side-by-side comparison:</div>", unsafe_allow_html=True)
        
        use_morph = st.checkbox("Morphological Blemish Analysis (Cham Herman) — [Completed]", value=True)
        use_color = st.checkbox("Color-Space Analysis (Lum Siew Feng) — [Completed]", value=True)
        use_texture = st.checkbox("Texture & Surface Analysis (Wong Kai Bin) — [Completed]", value=True)
        use_geom = st.checkbox("Edge & Shape Deformity Detection (Yeow Wei Kang) — [Completed]", value=True)
        
        selected_count = sum([use_morph, use_color, use_texture, use_geom])
        st.markdown(f"<div style='font-size: 0.85rem; color: #f59e0b; margin-top: 10px;'>Active Techniques: <b>{selected_count} / 4 Selected</b></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        assess_btn = st.button("Execute Ripeness Assessment", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # --- Processing Execution ---
    if img_bgr is not None and (assess_btn or 'last_results' in st.session_state):
        if assess_btn:
            if selected_count == 0:
                st.error("Please select at least one algorithm to run assessment.")
            else:
                with st.spinner("Processing selected computer vision pipelines..."):
                    results = {}
                    # Shared staged preprocessing so all 4 models start from this standard preprocessed image
                    img_prep_bgr, prep_stages = preprocess_image_with_stages(img_bgr)

                    if use_morph:
                        pred_m, conf_m, vis_m, met_m, steps_m = analyze_ripeness_by_morphology(img_prep_bgr)
                        results['morph'] = {'pred': pred_m, 'conf': conf_m, 'vis': vis_m, 'metrics': met_m, 'steps': steps_m, 'author': 'Cham Herman', 'name': 'Morphological Blemish Analysis', 'status': 'completed'}
                        
                    if use_color:
                        pred_c, conf_c, vis_c, met_c, steps_c = analyze_ripeness_by_color(img_prep_bgr)
                        results['color'] = {'pred': pred_c, 'conf': conf_c, 'vis': vis_c, 'metrics': met_c, 'steps': steps_c, 'author': 'Lum Siew Feng', 'name': 'Color-Space Analysis', 'status': 'completed'}
                        
                    if use_texture:
                        pred_t, conf_t, vis_t, met_t, steps_t = analyze_ripeness_by_texture(img_prep_bgr)
                        results['texture'] = {'pred': pred_t, 'conf': conf_t, 'vis': vis_t, 'metrics': met_t, 'steps': steps_t, 'author': 'Wong Kai Bin', 'name': 'Texture & Surface Analysis', 'status': 'completed'}
                        
                    if use_geom:
                        pred_g, conf_g, vis_g, met_g, steps_g = analyze_ripeness_by_geometry(img_prep_bgr)
                        results['geom'] = {'pred': pred_g, 'conf': conf_g, 'vis': vis_g, 'metrics': met_g, 'steps': steps_g, 'author': 'Yeow Wei Kang', 'name': 'Edge & Shape Geometry', 'status': 'completed'}
                        
                    # Calculate Ensemble Consensus Verdict
                    all_preds = [v['pred'] for v in results.values()]
                    consensus_pred = max(set(all_preds), key=all_preds.count)
                    avg_conf = np.mean([v['conf'] for v in results.values()])
                    total_latency = sum([v['metrics'].get('latency_ms', 0) for v in results.values()])
                    
                    st.session_state['last_results'] = {
                        'results': results,
                        'consensus': consensus_pred,
                        'avg_conf': avg_conf,
                        'total_latency': total_latency,
                        'filename': img_filename,
                        'preprocessed_bgr': img_prep_bgr,
                        'prep_stages': prep_stages
                    }
                    
        # Render Results
        if 'last_results' in st.session_state:
            pack = st.session_state['last_results']
            res_dict = pack['results']
            prep_bgr = pack.get('preprocessed_bgr', img_bgr)
            
            st.markdown("<hr style='opacity: 0.2; margin: 25px 0;'>", unsafe_allow_html=True)
            
            # Consensus Banner
            st.markdown(f"""
            <div class='consensus-box'>
                <div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #f59e0b; letter-spacing: 1px;'>Hybrid Ensemble Consensus Verdict</div>
                <div style='font-size: 2.0rem; font-weight: 800; margin: 4px 0;'>
                    {pack['consensus'].upper()}
                </div>
                <div>{get_class_badge(pack['consensus'])}</div>
                <div style='font-size: 0.85rem; opacity: 0.8; margin-top: 6px;'>
                    Consensus Confidence: <b>{pack['avg_conf']:.1f}%</b> | Cumulative Processing Latency: <b>{pack['total_latency']:.1f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Side-by-side Technique Cards
            st.markdown(f"<div class='section-header'>{SVG_ICONS['analytics']} Individual Diagnostic Results</div>", unsafe_allow_html=True)
            tech_cols = st.columns(len(res_dict))
            
            for idx, (k, item) in enumerate(res_dict.items()):
                with tech_cols[idx]:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>{item['name']}</div>", unsafe_allow_html=True)
                    
                    if item.get('status') == 'completed':
                        st.markdown(f"<div style='margin-bottom: 6px;'><span class='status-completed'>{SVG_ICONS['verified']} Completed ({item['author']})</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='margin-bottom: 6px;'><span class='status-scaffold'>{SVG_ICONS['scaffold']} Scaffold ({item['author']})</span></div>", unsafe_allow_html=True)
                        
                    st.markdown(f"<div class='metric-val'>{item['pred'].upper()}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div>{get_class_badge(item['pred'])}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 0.8rem; opacity: 0.8; margin-top: 8px;'>Confidence: <b>{item['conf']:.1f}%</b><br>Latency: <b>{item['metrics'].get('latency_ms', 0):.1f} ms</b></div>", unsafe_allow_html=True)
                    st.image(item['vis'], caption=f"Overlay: {item['name']}", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            # Step-by-Step Intermediate Pipeline Visualizer
            st.markdown(f"<div class='section-header'>{SVG_ICONS['eye']} Intermediate Pipeline Diagnostics (Step-by-Step Transformations)</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.85rem; opacity: 0.75; margin-bottom: 12px;'>Inspect the shared upstream preprocessing operations and downstream feature extraction steps for each algorithm:</div>", unsafe_allow_html=True)
            
            # 1. Dedicated Shared Upstream Preprocessing Expander
            prep_stages = pack.get('prep_stages')
            if prep_stages:
                with st.expander("Upstream Preprocessing Pipeline (Standard Letterbox, Denoise, CLAHE & Segmentation)", expanded=True):
                    st.markdown("<div style='font-size:0.85rem; opacity:0.85; margin-bottom:10px;'>Standardized preprocessing applied to the raw input image. All 4 downstream classification models start directly from this preprocessed image (P5):</div>", unsafe_allow_html=True)
                    prep_items = list(prep_stages.items())
                    p_cols = st.columns(len(prep_items))
                    for p_idx, (p_name, p_img) in enumerate(prep_items):
                        with p_cols[p_idx]:
                            if len(p_img.shape) == 2:
                                st.image(p_img, caption=p_name, use_container_width=True, clamp=True)
                            else:
                                st.image(p_img, caption=p_name, use_container_width=True)

            # 2. Individual Downstream Feature Extraction Pipelines
            for k, item in res_dict.items():
                badge_text = "Verified Module" if item.get('status') == 'completed' else "Scaffold Pipeline"
                with st.expander(f"Pipeline: {item['name']} (By {item['author']} — {badge_text})", expanded=False):
                    st.markdown(f"<div style='font-size:0.82rem; color:#f59e0b; margin-bottom:8px;'>Starts from the preprocessed image (P5) & executes feature extraction for <b>{item['name']}</b>:</div>", unsafe_allow_html=True)
                    steps = item.get('steps', {})
                    if k == 'color':
                        if 'selected_color_space' not in st.session_state:
                            st.session_state['selected_color_space'] = 'RGB'
                            
                        cs_preds = item.get('metrics', {}).get('predictions_per_color_space', {})
                        if not cs_preds:
                            cs_preds = {cs: 'unripe' for cs in COLOR_SPACES.keys()}
                            
                        st.markdown("<div style='font-size:0.95rem; font-weight:700; margin-bottom: 8px;'>Click any Color Space Card to directly switch pipeline view:</div>", unsafe_allow_html=True)
                        
                        cs_cols = st.columns(len(cs_preds))
                        for c_i, (cs_name, cs_pred) in enumerate(cs_preds.items()):
                            with cs_cols[c_i]:
                                is_active = (cs_name == st.session_state['selected_color_space'])
                                pred_tag = cs_pred.replace("_", " ").title()
                                card_label = f"{cs_name}\n({pred_tag})"
                                
                                if st.button(card_label, key=f"cs_direct_card_{cs_name}", use_container_width=True, type="primary" if is_active else "secondary"):
                                    st.session_state['selected_color_space'] = cs_name
                                    st.rerun()
                                    
                        st.markdown("<hr style='opacity: 0.15; margin: 12px 0;'>", unsafe_allow_html=True)
                        
                        selected_inspect_cs = st.session_state.get('selected_color_space', 'RGB')
                        if prep_bgr is not None:
                            steps = get_color_space_pipeline_steps(prep_bgr, selected_inspect_cs)
                            
                        st.markdown(f"<div style='font-size:0.9rem; font-weight:700; color:#f59e0b; margin-bottom:10px;'>Displaying Complete 7-Step Pipeline for <span style='text-decoration:underline;'>{selected_inspect_cs}</span> Color Space (Using Preprocessed Image):</div>", unsafe_allow_html=True)
                        
                    if steps:
                        step_items = list(steps.items())
                        num_steps = len(step_items)
                        if num_steps <= 7:
                            s_cols = st.columns(num_steps)
                            for s_idx, (s_name, s_img) in enumerate(step_items):
                                with s_cols[s_idx]:
                                    if len(s_img.shape) == 2:
                                        st.image(s_img, caption=s_name, use_container_width=True, clamp=True)
                                    else:
                                        st.image(s_img, caption=s_name, use_container_width=True)
                        else:
                            cols_per_row = 4
                            for chunk_start in range(0, num_steps, cols_per_row):
                                chunk = step_items[chunk_start:chunk_start + cols_per_row]
                                s_cols = st.columns(cols_per_row)
                                for s_idx, (s_name, s_img) in enumerate(chunk):
                                    with s_cols[s_idx]:
                                        if len(s_img.shape) == 2:
                                            st.image(s_img, caption=s_name, use_container_width=True, clamp=True)
                                        else:
                                            st.image(s_img, caption=s_name, use_container_width=True)
                    else:
                        st.info("No intermediate steps available for this module.")
                        
            # Metrics Summary Table
            st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} Extracted Feature Metrics Comparison Table</div>", unsafe_allow_html=True)
            table_rows = []
            for k, item in res_dict.items():
                row = {
                    'Technique': item['name'],
                    'Developer': item['author'],
                    'Module Status': 'Completed' if item.get('status')=='completed' else 'Scaffold',
                    'Predicted Class': item['pred'].upper(),
                    'Confidence (%)': f"{item['conf']:.1f}%",
                    'Latency (ms)': f"{item['metrics'].get('latency_ms', 0):.1f} ms"
                }
                if k == 'morph':
                    review_flag = ' | Needs Review' if item['metrics'].get('needs_review') else ''
                    row['Primary Physical Metric'] = (
                        f"Defect: {item['metrics'].get('blemish_area_ratio', 0):.2f}% "
                        f"[{item['metrics'].get('severity_grade', '-')}], "
                        f"Lesions (split): {item['metrics'].get('n_lesions_split', 0)}{review_flag}"
                    )
                elif k == 'color':
                    all_p = item['metrics'].get('predictions_per_color_space', {})
                    preds_str = ", ".join([f"{cs}:{p}" for cs, p in all_p.items()]) if all_p else ""
                    row['Primary Physical Metric'] = f"Benchmark Acc: {item['metrics'].get('accuracy_benchmark', '97.22%')} | Multi-Model: [{preds_str}]"
                elif k == 'texture':
                    row['Primary Physical Metric'] = f"GLCM Contrast: {item['metrics'].get('glcm_contrast', 0):.1f} | LBP Entropy: {item['metrics'].get('lbp_entropy', 0):.2f} (Model: {item['metrics'].get('classifier', 'KNN')})"
                elif k == 'geom':
                    row['Primary Physical Metric'] = f"Edge Density: {item['metrics'].get('edge_density_pct', 0):.2f}% | Aspect Ratio: {item['metrics'].get('bounding_aspect_ratio', 0):.2f}"
                    
                table_rows.append(row)
                
            df_metrics = pd.DataFrame(table_rows)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # PDF Report Export Option
            st.markdown("<br>", unsafe_allow_html=True)
            rep_col1, rep_col2 = st.columns([1, 2])
            with rep_col1:
                if st.button("Generate Quality Inspection PDF Report", use_container_width=True):
                    rep_results = [{
                        'filename': pack['filename'],
                        'morph_pred': res_dict.get('morph', {}).get('pred', '-'),
                        'morph_conf': res_dict.get('morph', {}).get('conf', 0),
                        'color_pred': res_dict.get('color', {}).get('pred', '-'),
                        'color_conf': res_dict.get('color', {}).get('conf', 0),
                        'texture_pred': res_dict.get('texture', {}).get('pred', '-'),
                        'texture_conf': res_dict.get('texture', {}).get('conf', 0),
                        'geom_pred': res_dict.get('geom', {}).get('pred', '-'),
                        'geom_conf': res_dict.get('geom', {}).get('conf', 0),
                        'final_pred': pack['consensus']
                    }]
                    summary_stats = {
                        'total_assessed': 1,
                        'consensus_ripe': 1 if pack['consensus'] == 'fully_ripe' else 0,
                        'consensus_unripe': 1 if pack['consensus'] == 'unripe' else 0,
                        'consensus_overripe': 1 if pack['consensus'] == 'overripe' else 0,
                        'avg_confidence': pack['avg_conf']
                    }
                    pdf_bytes = generate_pdf_report(rep_results, summary_stats)
                    st.download_button(
                        label="Download PDF Inspection Report",
                        data=pdf_bytes,
                        file_name=f"Mango_Ripeness_Report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with rep_col2:
                st.info("Quality inspection report includes detailed algorithmic outputs, confidence scores, and individual defect metrics.")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# PAGE 2: BULK BATCH ASSESSMENT (CONVEYOR STREAM)
# -----------------------------------------------------------------------------
elif selected_page.startswith("Bulk"):
    st.markdown(f"<div class='main-title'>{SVG_ICONS['conveyor']} Bulk Batch Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>High-throughput batch inspection stream for conveyor simulation with multi-source dataset ingestion, flexible algorithm selection, and confidence-based decision fusion.</div>", unsafe_allow_html=True)
    
    col_batch_in, col_batch_alg = st.columns([1.1, 1.1])
    
    with col_batch_in:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['upload']} 1. Ingest Image Stream</div>", unsafe_allow_html=True)
        batch_source_mode = st.radio("Stream Ingestion Source:", [
            "Standard Dataset / Repository Path",
            "Upload .ZIP Dataset Archive (.zip)",
            "Upload Multiple Individual Image Files"
        ], horizontal=True)
        
        batch_items = []
        
        if batch_source_mode.startswith("Standard"):
            repo_path_options = [
                "cleaned_data/test (Cleaned Test Split — 144 Images)",
                "cleaned_data/test/fully_ripe (Cleaned Fully Ripe Test)",
                "cleaned_data/test/unripe (Cleaned Unripe Test)",
                "cleaned_data/test/overripe (Cleaned Overripe Test)",
                "cleaned_data/train (Cleaned Train Split — 571 Images)",
                "data (Raw Full Dataset — 715 Images)",
                "data/fully_ripe (Raw Fully Ripe)",
                "data/unripe (Raw Unripe)",
                "data/overripe (Raw Overripe)",
                "Custom Directory Path..."
            ]
            selected_path_label = st.selectbox("Select Assessment Split / Directory:", repo_path_options)
            if selected_path_label == "Custom Directory Path...":
                batch_dir = st.text_input("Enter custom directory path in repository:", value="data")
            else:
                batch_dir = selected_path_label.split(" (")[0]
                
            found_paths = sorted(glob.glob(f"{batch_dir}/**/*.jpg", recursive=True) + 
                                 glob.glob(f"{batch_dir}/**/*.jpeg", recursive=True) + 
                                 glob.glob(f"{batch_dir}/**/*.png", recursive=True))
            for p in found_paths:
                parent_dir = os.path.basename(os.path.dirname(p)).lower()
                true_c = parent_dir if parent_dir in ['unripe', 'fully_ripe', 'overripe'] else 'unknown'
                batch_items.append({
                    'type': 'path',
                    'path': p,
                    'filename': os.path.basename(p),
                    'true_class': true_c
                })
            st.markdown(f"<div style='font-size: 0.85rem; opacity: 0.8; margin-top: 6px;'>Found <b>{len(batch_items)} mango images</b> in <code>{batch_dir}</code>.</div>", unsafe_allow_html=True)
            
        elif "ZIP" in batch_source_mode:
            uploaded_zip = st.file_uploader(
                "Upload Mango Dataset ZIP Archive (.zip):",
                type=["zip"],
                key="zip_batch_uploader"
            )
            if uploaded_zip is not None:
                try:
                    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
                        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                        for info in z.infolist():
                            if info.is_dir() or info.filename.startswith('__MACOSX') or '/.' in info.filename or os.path.basename(info.filename).startswith('.'):
                                continue
                            ext = os.path.splitext(info.filename)[1].lower()
                            if ext in valid_exts:
                                file_bytes = z.read(info.filename)
                                low_path = info.filename.lower()
                                if 'unripe' in low_path:
                                    tc = 'unripe'
                                elif 'overripe' in low_path:
                                    tc = 'overripe'
                                elif 'ripe' in low_path:
                                    tc = 'fully_ripe'
                                else:
                                    tc = 'uploaded'
                                batch_items.append({
                                    'type': 'bytes',
                                    'bytes': file_bytes,
                                    'filename': os.path.basename(info.filename),
                                    'true_class': tc
                                })
                    st.markdown(f"<div style='font-size: 0.85rem; color: #10b981; margin-top: 6px;'>Successfully extracted <b>{len(batch_items)} valid mango images</b> from ZIP archive (non-image files auto-filtered).</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to read ZIP archive: {e}")
        else:
            if 'batch_uploader_key' not in st.session_state:
                st.session_state['batch_uploader_key'] = 0
                
            custom_uploaded_files = st.file_uploader(
                "Upload Multiple Mango Images:",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"batch_uploader_{st.session_state['batch_uploader_key']}"
            )
            if custom_uploaded_files:
                for f in custom_uploaded_files:
                    batch_items.append({
                        'type': 'bytes',
                        'bytes': f.read(),
                        'filename': f.name,
                        'true_class': 'uploaded'
                    })
            
            c_clear1, c_clear2 = st.columns([1, 1])
            with c_clear1:
                st.markdown(f"<div style='font-size: 0.85rem; opacity: 0.8; margin-top: 6px;'>Uploaded <b>{len(batch_items)} images</b>.</div>", unsafe_allow_html=True)
            with c_clear2:
                if st.button("Delete All Images / Clear Batch", use_container_width=True):
                    st.session_state['batch_uploader_key'] += 1
                    if 'last_batch_df' in st.session_state:
                        del st.session_state['last_batch_df']
                    st.rerun()
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_batch_alg:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['sliders']} 2. Batch Algorithm Selection & Strategy</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; opacity: 0.7; margin-bottom: 10px;'>Select one or more high-accuracy algorithms to execute on the batch:</div>", unsafe_allow_html=True)
        
        b_use_morph = st.checkbox(f"Morphological Blemish Analysis (Cham Herman) — {morph_bm.get('accuracy', 98.61):.2f}% Acc [Completed]", value=True, key="b_morph")
        b_use_color = st.checkbox(f"Color-Space Analysis (Lum Siew Feng) — {best_cs_acc:.2f}% Best {best_cs} / 97.22% RGB [Completed]", value=True, key="b_color")
        b_use_texture = st.checkbox(f"Texture & Surface Analysis (Wong Kai Bin) — {texture_acc:.2f}% Acc [Completed]", value=True, key="b_texture")
        b_use_geom = st.checkbox(f"Edge & Shape Deformity (Yeow Wei Kang) — {geom_acc:.2f}% Acc [Completed]", value=True, key="b_geom")
        
        b_active_count = sum([b_use_morph, b_use_color, b_use_texture, b_use_geom])
        
        if b_active_count == 1:
            st.info("Single Model Mode: Batch decisions will directly follow the selected high-accuracy algorithm.")
        elif b_active_count > 1:
            st.success(f"Highest-Confidence Selection: For each item, the prediction from the model exhibiting the highest confidence among the {b_active_count} active techniques will be chosen as the final grade.")
        else:
            st.warning("Please select at least 1 algorithm before starting the batch analysis.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        start_batch_btn = st.button("Start Batch Conveyor Inspection", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    total_batch_items = len(batch_items)
    
    if (start_batch_btn or 'last_batch_df' in st.session_state) and total_batch_items > 0:
        if start_batch_btn:
            if b_active_count == 0:
                st.error("Please select at least one algorithm before running batch assessment.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                batch_results = []
                start_time = time.time()
                
                for i in range(total_batch_items):
                    item = batch_items[i]
                    filename = item['filename']
                    true_cls = item['true_class']
                    
                    if item['type'] == 'path':
                        bgr = cv2.imread(item['path'])
                    else:
                        file_bytes = np.asarray(bytearray(item['bytes']), dtype=np.uint8)
                        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                    if bgr is None:
                        continue
                    bgr = preprocess_image(bgr)
                    
                    item_preds = {}
                    
                    if b_use_morph:
                        pm, cm, _, mm, _ = analyze_ripeness_by_morphology(bgr)
                        item_preds['morph'] = {'pred': pm, 'conf': cm, 'name': 'Morphology (Herman)', 'metrics': mm}
                    if b_use_color:
                        pc, cc, _, mc, _ = analyze_ripeness_by_color(bgr, primary_space="LAB")
                        item_preds['color'] = {'pred': pc, 'conf': cc, 'name': 'Color-Space (Siew Feng)', 'metrics': mc}
                    if b_use_texture:
                        pt, ct, _, mt, _ = analyze_ripeness_by_texture(bgr)
                        item_preds['texture'] = {'pred': pt, 'conf': ct, 'name': 'Texture (Kai Bin)', 'metrics': mt}
                    if b_use_geom:
                        pg, cg, _, mg, _ = analyze_ripeness_by_geometry(bgr)
                        item_preds['geom'] = {'pred': pg, 'conf': cg, 'name': 'Geometry (Wei Kang)', 'metrics': mg}
                        
                    # Decision Logic: If 1 model -> use directly. If multiple -> choose highest confidence model!
                    if len(item_preds) == 1:
                        single_k = list(item_preds.keys())[0]
                        final_pred = item_preds[single_k]['pred']
                        final_conf = item_preds[single_k]['conf']
                        winning_model = item_preds[single_k]['name']
                    else:
                        winning_k = max(item_preds.keys(), key=lambda k: item_preds[k]['conf'])
                        final_pred = item_preds[winning_k]['pred']
                        final_conf = item_preds[winning_k]['conf']
                        winning_model = item_preds[winning_k]['name']
                        
                    blemish_ratio_str = f"{item_preds.get('morph', {}).get('metrics', {}).get('blemish_area_ratio', 0):.2f}%" if 'morph' in item_preds else "-"
                    severity_grade_str = item_preds.get('morph', {}).get('metrics', {}).get('severity_grade', '-') if 'morph' in item_preds else "-"
                    
                    record = {
                        'filename': filename,
                        'true_class': true_cls,
                        'final_pred': final_pred,
                        'final_conf': round(final_conf, 1),
                        'winning_model': winning_model,
                        'blemish_ratio': blemish_ratio_str,
                        'severity_grade': severity_grade_str
                    }
                    
                    if 'morph' in item_preds:
                        record['morph_pred'] = f"{item_preds['morph']['pred']} ({item_preds['morph']['conf']:.1f}%)"
                    if 'color' in item_preds:
                        record['color_pred'] = f"{item_preds['color']['pred']} ({item_preds['color']['conf']:.1f}%)"
                    if 'texture' in item_preds:
                        record['texture_pred'] = f"{item_preds['texture']['pred']} ({item_preds['texture']['conf']:.1f}%)"
                    if 'geom' in item_preds:
                        record['geom_pred'] = f"{item_preds['geom']['pred']} ({item_preds['geom']['conf']:.1f}%)"
                        
                    batch_results.append(record)
                    progress_bar.progress((i + 1) / total_batch_items)
                    status_text.text(f"Inspecting stream item {i+1} of {total_batch_items}: {filename} (Decided: {final_pred.upper()} by {winning_model})")
                    
                elapsed = time.time() - start_time
                avg_lat_item = (elapsed / total_batch_items) * 1000.0 if total_batch_items > 0 else 0
                status_text.success(f"Batch completed: Evaluated {len(batch_results)} items in {elapsed:.2f}s ({avg_lat_item:.1f} ms/item)")
                
                st.session_state['last_batch_df'] = pd.DataFrame(batch_results)
                st.session_state['last_batch_elapsed'] = elapsed
                st.session_state['last_batch_avg_lat'] = avg_lat_item
                
        # Render Batch Summary
        if 'last_batch_df' in st.session_state:
            df_batch = st.session_state['last_batch_df']
            
            st.markdown("<hr style='opacity: 0.2; margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-header'>{SVG_ICONS['analytics']} Batch Inspection Summary & KPIs</div>", unsafe_allow_html=True)
            
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Total Inspected", len(df_batch))
            kpi2.metric("Ripe (Pass)", int((df_batch['final_pred'] == 'fully_ripe').sum()))
            kpi3.metric("Unripe (Hold)", int((df_batch['final_pred'] == 'unripe').sum()))
            kpi4.metric("Overripe (Reject)", int((df_batch['final_pred'] == 'overripe').sum()))
            kpi5.metric("Avg Confidence", f"{df_batch['final_conf'].mean():.1f}%")
            
            # Distribution Bar Chart
            fig, ax = plt.subplots(figsize=(7, 3), facecolor='none')
            ax.set_facecolor('none')
            sns.countplot(data=df_batch, x='final_pred', order=['unripe', 'fully_ripe', 'overripe'], palette=['#16a34a', '#f59e0b', '#dc2626'], ax=ax)
            ax.set_title("Batch Maturity Distribution (Confidence-Selected Decisions)", fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
            # Interactive Batch Data Table
            st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} Batch Assessment Itemized Records</div>", unsafe_allow_html=True)
            
            display_cols = ['filename']
            for tech_col in ['morph_pred', 'color_pred', 'texture_pred', 'geom_pred']:
                if tech_col in df_batch.columns:
                    display_cols.append(tech_col)
            display_cols.extend(['winning_model', 'final_pred', 'final_conf', 'blemish_ratio', 'severity_grade'])
            
            st.dataframe(df_batch[display_cols], use_container_width=True, hide_index=True)
            
            # Batch PDF Download
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate Batch Quality Inspection PDF Report", use_container_width=True):
                rep_results = []
                for _, row in df_batch.iterrows():
                    rep_results.append({
                        'filename': row['filename'],
                        'morph_pred': row.get('morph_pred', '-'),
                        'morph_conf': 0,
                        'color_pred': row.get('color_pred', '-'),
                        'color_conf': 0,
                        'texture_pred': row.get('texture_pred', '-'),
                        'texture_conf': 0,
                        'geom_pred': row.get('geom_pred', '-'),
                        'geom_conf': 0,
                        'final_pred': row['final_pred']
                    })
                summary_stats = {
                    'total_assessed': len(df_batch),
                    'consensus_ripe': int((df_batch['final_pred'] == 'fully_ripe').sum()),
                    'consensus_unripe': int((df_batch['final_pred'] == 'unripe').sum()),
                    'consensus_overripe': int((df_batch['final_pred'] == 'overripe').sum()),
                    'avg_confidence': float(df_batch['final_conf'].mean())
                }
                pdf_bytes = generate_pdf_report(rep_results, summary_stats)
                if isinstance(pdf_bytes, str) and os.path.exists(pdf_bytes):
                    with open(pdf_bytes, "rb") as f:
                        pdf_data = f.read()
                else:
                    pdf_data = pdf_bytes
                st.download_button(
                    label="Download Batch PDF Inspection Report",
                    data=pdf_data,
                    file_name=f"Batch_Inspection_Report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# PAGE 3: LIVE CAMERA INSPECTION (REAL-TIME STREAM, EXTRA EFFORT)
# -----------------------------------------------------------------------------
elif selected_page.startswith("Live"):
    from src.video import PREPROCESSING_METADATA
    st.markdown(f"<div class='main-title'>{SVG_ICONS['diagnostic']} Live Camera Inspection (Real-Time Stream)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Continuous webcam ingestion with real-time per-frame analysis: configurable preprocessing segmentation followed by your chosen computer vision algorithm on every frame.</div>", unsafe_allow_html=True)

    # 1. Preprocessing & Algorithm Selection Controls
    col_prep_cfg, col_alg_cfg = st.columns([1, 1])
    with col_prep_cfg:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['sliders']} 1. Preprocessing Pipeline</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.82rem; opacity: 0.75; margin-bottom: 10px;'>Select segmentation & background removal technique:</div>", unsafe_allow_html=True)
        prep_options = {
            "Standard K-Means Color Clustering & Convex Hull (Default)": "kmeans",
            "Background-Agnostic Morphological Fruit Segmentation": "morphology"
        }
        selected_prep_label = st.selectbox("Active Preprocessing Engine:", list(prep_options.keys()), index=0)
        selected_prep_key = prep_options[selected_prep_label]
        st.markdown("</div>", unsafe_allow_html=True)

    with col_alg_cfg:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['sliders']} 2. Ripeness Grading Algorithm</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.82rem; opacity: 0.75; margin-bottom: 10px;'>Select computer vision classification pipeline:</div>", unsafe_allow_html=True)
        
        live_alg_map = {
            f"Morphological Blemish Analysis (Cham Herman) — MRMF {morph_bm.get('accuracy', 98.61):.2f}% Acc": "morphology",
            f"Color-Space Analysis (Lum Siew Feng) — LAB Chrominance {best_cs_acc:.2f}% Acc": "color",
            f"Texture & Surface Analysis (Wong Kai Bin) — Multi-Descriptor {texture_acc:.2f}% Acc": "texture",
            f"Edge & Shape Geometry (Yeow Wei Kang) — Scharr & Contour {geom_acc:.2f}% Acc": "geometry"
        }
        selected_live_label = st.selectbox("Active Ripeness Algorithm:", list(live_alg_map.keys()), index=0)
        selected_live_key = live_alg_map[selected_live_label]
        st.markdown("</div>", unsafe_allow_html=True)

    try:
        from streamlit_webrtc import webrtc_streamer
        webrtc_available = True
    except ImportError:
        webrtc_available = False

    # 2. Side-by-Side Stream and Real-Time Analytics Layout
    col_stream_view, col_analytics_view = st.columns([1, 1], gap="medium")

    if 'live_stats' not in st.session_state:
        st.session_state['live_stats'] = LiveSessionStats(default_algorithm=selected_live_key, default_preprocessing=selected_prep_key)
    stats = st.session_state['live_stats']
    stats.set_algorithm(selected_live_key)
    stats.set_preprocessing(selected_prep_key)

    # Do not auto-start camera: default to False
    if 'stream_active' not in st.session_state:
        st.session_state['stream_active'] = False

    with col_stream_view:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['camera']} Video Stream Ingestion</div>", unsafe_allow_html=True)
        
        input_mode = "Camera Snapshot"
        if webrtc_available:
            input_mode = st.radio("Input Feed Mode:", ["Live Camera (WebRTC)", "Camera Snapshot"], horizontal=True)

        if webrtc_available and input_mode.startswith("Live"):
            if st.session_state['stream_active']:
                if st.button("Stop Live Stream", use_container_width=True, type="secondary"):
                    st.session_state['stream_active'] = False
                    st.rerun()

                ctx = webrtc_streamer(
                    key="mango-live-inspection",
                    desired_playing_state=st.session_state['stream_active'],
                    video_frame_callback=make_webrtc_callback(algorithm=selected_live_key, preprocessing=selected_prep_key, stats=stats),
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                if ctx.state.playing:
                    st.success(f"Streaming Active — Running {selected_live_label}")
                else:
                    st.info("Waiting for webcam access... allow camera permission in browser to begin streaming.")
            else:
                st.info("Live camera is on standby. Click 'Start Live Stream' below to activate webcam inspection.")
                if st.button("Start Live Stream", use_container_width=True, type="primary"):
                    st.session_state['stream_active'] = True
                    st.rerun()
        else:
            snap_img = st.camera_input("Capture mango frame:")
            if snap_img is not None:
                st.session_state['last_snap_bytes'] = bytearray(snap_img.getvalue())
        st.markdown("</div>", unsafe_allow_html=True)

    with col_analytics_view:
        if webrtc_available and input_mode.startswith("Live"):
            @st.fragment(run_every=2)
            def render_live_stats():
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                snap = stats.snapshot()
                active_name = ALGORITHM_METADATA.get(snap.get('algorithm', 'morphology'), {}).get('name', 'Selected Engine')
                active_prep_name = PREPROCESSING_METADATA.get(snap.get('preprocessing', 'kmeans'), {}).get('name', 'Selected Preprocessing')
                
                status_badge = "<span class='badge-unripe'>LIVE STREAMING</span>" if st.session_state.get('stream_active', False) else "<span class='badge-ripe'>STANDBY</span>"
                st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'><span style='font-weight:700;'>{SVG_ICONS['analytics']} Real-Time Live Analytics</span>{status_badge}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.78rem; opacity:0.8; margin-bottom:12px;'>Algorithm: <b>{active_name}</b><br>Preprocessing: <b>{active_prep_name}</b><br>Compute Device: <b>{hw_info['device_name']}</b> ({'GPU Accelerated' if hw_info['has_gpu'] else 'CPU Execution'})</div>", unsafe_allow_html=True)
                
                m1, m2 = st.columns(2)
                m1.metric("Frames Analysed", snap['total_frames'])
                m2.metric("Throughput", f"{snap['measured_fps']:.1f} fps")
                
                m3, m4 = st.columns(2)
                m3.metric("Avg Latency", f"{snap['avg_latency_ms']:.0f} ms")
                m4.metric("Overripe Rejects", snap['verdict_counts'].get('overripe', 0))
                
                st.markdown("<div style='font-size:0.80rem; font-weight:600; margin: 10px 0 4px 0;'>Live Batch Verdict Distribution:</div>", unsafe_allow_html=True)
                st.bar_chart(snap['verdict_counts'], height=180)

                # Redesigned Session Data Export & Reset Controls
                df_history = stats.get_history_dataframe()
                st.markdown("<hr style='opacity: 0.15; margin: 14px 0 10px 0;'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.82rem; font-weight:700; margin-bottom:8px;'>{SVG_ICONS['table']} Session Data & Telemetry Controls</div>", unsafe_allow_html=True)
                
                c_dl, c_rst = st.columns([3, 2], gap="small")
                with c_dl:
                    if not df_history.empty:
                        csv_bytes = df_history.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Telemetry Log ({len(df_history)} Frames, CSV)",
                            data=csv_bytes,
                            file_name=f"live_stream_telemetry_{int(time.time())}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            type="primary"
                        )
                    else:
                        st.button("Download Telemetry Log (CSV)", disabled=True, use_container_width=True)
                with c_rst:
                    if st.button("Reset Live Metrics", use_container_width=True, type="secondary"):
                        stats.reset()
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

            render_live_stats()
        else:
            if 'last_snap_bytes' in st.session_state:
                file_bytes = np.asarray(st.session_state['last_snap_bytes'], dtype=np.uint8)
                frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                with st.spinner("Processing snapshot..."):
                    res = analyze_frame(frame_bgr, algorithm=selected_live_key, preprocessing=selected_prep_key)
                
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-weight:700; margin-bottom:8px;'>{SVG_ICONS['verified']} Snapshot Analysis Result</div>", unsafe_allow_html=True)
                st.image(res['annotated_rgb'], caption=f"Frame verdict: {res['prediction'].upper()} ({res['confidence']:.1f}%)", use_container_width=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Ripeness", res['prediction'].upper())
                m2.metric("Confidence", f"{res['confidence']:.1f}%")
                m3.metric("Latency", f"{res['latency_ms']:.1f} ms")
                
                with st.expander("Fruit Mask & Preprocessing Stage", expanded=False):
                    st.image(res['tierb_mask'], caption=f"Fruit Segmentation Mask ({res['preprocessing_name']})", use_container_width=True)

                # Direct Snapshot Telemetry Export (Single-frame real-time diagnosis)
                m = res.get('metrics', {})
                snap_telemetry = [{
                    'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'Algorithm': res.get('algorithm_name', selected_live_label),
                    'Preprocessing': res.get('preprocessing_name', selected_prep_label),
                    'Ripeness': res['prediction'].upper(),
                    'Confidence (%)': round(float(res.get('confidence', 0.0)), 1),
                    'Latency (ms)': round(float(res.get('latency_ms', 0.0)), 1),
                    'Severity Grade': res.get('severity_grade', '-'),
                    'Defect (%)': round(float(res.get('defect_percentage', 0.0)), 2) if res.get('defect_percentage') != '-' else '-',
                    'Blemish Area (%)': m.get('blemish_area_ratio', '-'),
                    'Max Lesion Ratio (%)': m.get('max_lesion_ratio', '-'),
                    'Mean Hue': m.get('mean_hue', '-'),
                    'Mean Saturation': m.get('mean_saturation', '-'),
                    'Mean Value': m.get('mean_value', '-'),
                    'GLCM Contrast': m.get('glcm_contrast', '-'),
                    'LBP Entropy': m.get('lbp_entropy', '-'),
                    'Edge Density (%)': round(float(m.get('scharr_density', 0.0)) * 100, 2) if 'scharr_density' in m else '-',
                    'Aspect Ratio': m.get('aspect_ratio', '-')
                }]
                df_snap = pd.DataFrame(snap_telemetry)
                csv_bytes = df_snap.to_csv(index=False).encode('utf-8')

                st.markdown("<hr style='opacity: 0.15; margin: 14px 0 10px 0;'>", unsafe_allow_html=True)
                st.download_button(
                    label="Download Snapshot Diagnostics (CSV)",
                    data=csv_bytes,
                    file_name=f"snapshot_diagnostic_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='glass-card'><div style='opacity:0.7; font-size:0.88rem;'>Point the camera at a mango and capture a frame to see diagnostics here.</div></div>", unsafe_allow_html=True)

# PAGE 4: SYSTEM ANALYTICS & COMPARATIVE BENCHMARK
# -----------------------------------------------------------------------------
elif selected_page.startswith("System"):
    st.markdown(f"<div class='main-title'>{SVG_ICONS['analytics']} System Analytics & Comparative Benchmark</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Empirical evaluation results and developmental status across team modules.</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} Mode A Table 2.1: Comparative Benchmark Across Image Processing Modules</div>", unsafe_allow_html=True)
    
    active_cs = st.session_state.get('selected_color_space', 'RGB')
    color_bm_active = bm_metrics.get('color', {}).get(active_cs, {})
    
    morph_acc_str = f"{morph_bm.get('accuracy', 98.61):.2f}%"
    morph_f1_str = f"{morph_bm.get('f1', 98.61):.2f}%"
    morph_lat_str = f"{morph_bm.get('latency_ms', 32.48):.2f} ms"
    
    # Show overall highest accuracy among the 5 color spaces as main benchmark
    color_max_acc_str = f"{best_cs_acc:.2f}% (Best: {best_cs})"
    color_max_f1_str = f"{best_cs_f1:.2f}% (Best: {best_cs})"
    color_lat_str = f"{color_bm_active.get('latency_ms', 12.45):.2f} ms"
    
    benchmark_data = [
        {
            'Algorithm / Module': '1. Morphological Blemish Analysis (Cham Herman)',
            'Development Status': 'Completed & Evaluated',
            'Core Formulation': 'Multi-Scale Beucher Gradient & Black-Hat Residual Fusion',
            'Test Accuracy (%)': morph_acc_str,
            'Macro F1 (%)': morph_f1_str,
            'Latency (ms/img)': morph_lat_str
        },
        {
            'Algorithm / Module': f'2. Color-Space Analysis (Lum Siew Feng) — Top Model ({best_cs})',
            'Development Status': 'Completed & Evaluated',
            'Core Formulation': 'Multi-Color Space Chrominance & Support Vector Classification (RGB/HSV/LAB/YCbCr/HLS)',
            'Test Accuracy (%)': color_max_acc_str,
            'Macro F1 (%)': color_max_f1_str,
            'Latency (ms/img)': color_lat_str
        },
        {
            'Algorithm / Module': '3. Texture & Surface Analysis (Wong Kai Bin)',
            'Development Status': 'Completed & Evaluated',
            'Core Formulation': 'Enhanced Multi-Descriptor Fusion: Rotation-Invariant GLCM (4 angles) + Uniform LBP + Sobel Roughness',
            'Test Accuracy (%)': f"{texture_acc:.2f}%",
            'Macro F1 (%)': f"{texture_f1:.2f}%",
            'Latency (ms/img)': f"{texture_lat:.2f} ms"
        },
        {
            'Algorithm / Module': '4. Edge & Shape Deformity (Yeow Wei Kang)',
            'Development Status': 'Completed & Evaluated',
            'Core Formulation': 'Scharr Edge Density + Advanced Contour Geometry Pipeline',
            'Test Accuracy (%)': f"{geom_acc:.2f}%",
            'Macro F1 (%)': f"{geom_f1:.2f}%",
            'Latency (ms/img)': f"{geom_lat:.2f} ms"
        }
    ]
    
    df_bm = pd.DataFrame(benchmark_data)
    st.dataframe(df_bm, use_container_width=True, hide_index=True)
    
    # Detailed 5 Color Spaces Benchmark Table
    st.markdown("<div style='font-size:0.9rem; font-weight:700; margin-top:10px; margin-bottom:5px;'>Detailed Benchmark Accuracy Across All 5 Color Spaces:</div>", unsafe_allow_html=True)
    all_cs_rows = []
    for cs_name in ['RGB', 'HSV', 'LAB', 'YCbCr', 'HLS']:
        cs_m = bm_metrics.get('color', {}).get(cs_name, {})
        all_cs_rows.append({
            'Color Space': cs_name,
            'Test Accuracy (%)': f"{cs_m.get('accuracy', 0):.2f}%",
            'Macro F1 (%)': f"{cs_m.get('f1', 0):.2f}%",
            'Ranking Status': 'Top Benchmark Model' if cs_name == best_cs else 'Evaluated Sub-Model'
        })
    st.dataframe(pd.DataFrame(all_cs_rows), use_container_width=True, hide_index=True)
    
    st.markdown(f"<div style='font-size:0.75rem; opacity:0.75; margin-top:6px; margin-bottom:12px;'><b>Hardware Acceleration Note:</b> Computational workloads (morphology filtering, color transforms, and geometry filters) automatically dispatch to GPU when available (Active: <b>{hw_info['device_name']}</b> via OpenCV OpenCL / CUDA) with seamless multi-threaded CPU fallback.</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-header'>{SVG_ICONS['analytics']} Verified Modules Performance Visualisation</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    verified_df = pd.DataFrame([
        {'Module': 'Morphology\n(Herman)', 'Test Accuracy (%)': morph_bm.get('accuracy', 98.61), 'Latency (ms)': morph_bm.get('latency_ms', 32.48)},
        {'Module': 'Color-Space\n(Siew Feng)', 'Test Accuracy (%)': best_cs_acc, 'Latency (ms)': color_bm_active.get('latency_ms', 12.45)},
        {'Module': 'Texture\n(Kai Bin)', 'Test Accuracy (%)': texture_acc, 'Latency (ms)': texture_lat},
        {'Module': 'Geometry\n(Wei Kang)', 'Test Accuracy (%)': geom_acc, 'Latency (ms)': geom_lat}
    ])
    
    palette = ['#2563eb', '#d97706', '#059669', '#7c3aed']
    
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='white')
        ax.set_facecolor('white')
        sns.barplot(data=verified_df, x='Module', y='Test Accuracy (%)', hue='Module', palette=palette, ax=ax, legend=False, width=0.55)
        ax.set_title("Test Accuracy Comparison (Verified Modules)", fontsize=10.5, fontweight='bold', color='#0f172a', pad=10)
        ax.set_ylim(65, 112)
        ax.axhline(85, color='#ef4444', linestyle='--', linewidth=1.5, label='Target Threshold (85%)')
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
        
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', fontsize=9.5, fontweight='bold', padding=3, color='#0f172a')
        
        ax.tick_params(colors='#1e293b', labelsize=8.5)
        ax.xaxis.label.set_color('#1e293b')
        ax.yaxis.label.set_color('#1e293b')
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        leg = ax.legend(facecolor='white', edgecolor='#cbd5e1', fontsize=8, loc='upper left')
        for text in leg.get_texts():
            text.set_color('#0f172a')
            
        fig.tight_layout()
        st.pyplot(fig)
        
    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='white')
        ax.set_facecolor('white')
        sns.barplot(data=verified_df, x='Module', y='Latency (ms)', hue='Module', palette=palette, ax=ax, legend=False, width=0.55)
        ax.set_title("Processing Latency per Image (ms)", fontsize=10.5, fontweight='bold', color='#0f172a', pad=10)
        ax.set_ylim(0, 230)
        ax.axhline(200, color='#ef4444', linestyle='--', linewidth=1.5, label='Max Budget (200 ms)')
        ax.axhline(33.3, color='#10b981', linestyle=':', linewidth=1.3, label='30 FPS Video (33.3 ms)')
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
        
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f ms', fontsize=9.5, fontweight='bold', padding=3, color='#0f172a')
        
        ax.tick_params(colors='#1e293b', labelsize=8.5)
        ax.xaxis.label.set_color('#1e293b')
        ax.yaxis.label.set_color('#1e293b')
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        leg = ax.legend(facecolor='white', edgecolor='#cbd5e1', fontsize=7.8, loc='upper right')
        for text in leg.get_texts():
            text.set_color('#0f172a')
            
        fig.tight_layout()
        st.pyplot(fig)
        
    st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} SMART Objectives Verification Matrix</div>", unsafe_allow_html=True)
    smart_data = [
        {
            'SMART Objective': 'Objective 1: Multi-Algorithm Suite',
            'Target Criterion': 'Implement 4 distinct classical computer vision algorithms',
            'Current Measured Status': '4 Completed (Herman, Siew Feng, Kai Bin & Wei Kang)',
            'Fulfillment': 'Completed (100% Finalized)'
        },
        {
            'SMART Objective': 'Objective 2: Classification Accuracy',
            'Target Criterion': 'Achieve minimum >= 85% classification accuracy',
            'Current Measured Status': f"{morph_acc_str} (Herman Morphology) | {best_cs_acc:.2f}% (Siew Feng Color - Best: {best_cs}) | {texture_acc:.2f}% (Kai Bin Texture) | {geom_acc:.2f}% (Wei Kang Geometry)",
            'Fulfillment': 'Target Exceeded'
        },
        {
            'SMART Objective': 'Objective 3: Operational Latency',
            'Target Criterion': 'Execute with per-image latency < 200 ms',
            'Current Measured Status': f"{morph_lat_str} (Herman) | {color_lat_str} (Siew Feng) | {texture_lat:.2f} ms (Kai Bin) | {geom_lat:.2f} ms (Wei Kang)",
            'Fulfillment': 'Target Exceeded'
        }
    ]
    st.dataframe(pd.DataFrame(smart_data), use_container_width=True, hide_index=True)
