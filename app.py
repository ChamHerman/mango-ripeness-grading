import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import modular image processing engines
from preprocessing import preprocess_image
from src.morphology import analyze_ripeness_by_morphology
from src.color_space import analyze_ripeness_by_color, get_color_space_pipeline_steps, COLOR_SPACES
from src.texture import analyze_ripeness_by_texture
from src.geometry import analyze_ripeness_by_geometry
from src.reports import generate_pdf_report

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
    'eye': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>'
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
     "System Analytics & Comparative Benchmark"]
)

st.sidebar.markdown("<br><hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style='font-size: 0.75rem; opacity: 0.8;'>
    <b>Team Module Status:</b><br><br>
    <div style='margin-bottom: 6px;'><b>Cham Herman</b>: Morphological Blemish<br><span class='status-completed'>{SVG_ICONS['verified']} Completed & Verified</span></div>
    <div style='margin-bottom: 6px;'><b>Lum Siew Feng</b>: Color-Space Analysis<br><span class='status-completed'>{SVG_ICONS['verified']} Completed & Verified (97.22% Acc)</span></div>
    <div style='margin-bottom: 6px;'><b>Wong Kai Bin</b>: Texture & Surface GLCM<br><span class='status-scaffold'>{SVG_ICONS['scaffold']} Scaffold (Pending Final Notebook)</span></div>
    <div><b>Yeow Wei Kang</b>: Edge & Deformity Detection<br><span class='status-scaffold'>{SVG_ICONS['scaffold']} Scaffold (Pending Final Notebook)</span></div>
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
                "Sample Unripe Mango (Stage 0)": "cleaned_data/test/unripe",
                "Sample Fully Ripe Mango (Stage 3)": "cleaned_data/test/fully_ripe",
                "Sample Overripe Mango (Necrotic Lesions)": "cleaned_data/test/overripe"
            }
            selected_sample_label = st.selectbox("Select Test Image:", list(sample_options.keys()))
            sample_dir = sample_options[selected_sample_label]
            available_samples = sorted(glob.glob(f"{sample_dir}/*.*"))
            if available_samples:
                img_path = available_samples[0]
                img_bgr = cv2.imread(img_path)
                img_filename = os.path.basename(img_path)
                
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Input Image: {img_filename}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_config:
        st.markdown(f"<div class='glass-card'><div style='font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>{SVG_ICONS['sliders']} 2. Algorithm Selection</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; opacity: 0.7; margin-bottom: 12px;'>Select which classical algorithms to execute for side-by-side comparison:</div>", unsafe_allow_html=True)
        
        use_morph = st.checkbox("Morphological Blemish Analysis (Cham Herman) — [Completed]", value=True)
        use_color = st.checkbox("Color-Space Analysis (Lum Siew Feng) — [Completed]", value=True)
        use_texture = st.checkbox("Texture & Surface GLCM Analysis (Wong Kai Bin) — [Scaffold]", value=False)
        use_geom = st.checkbox("Edge & Shape Deformity Detection (Yeow Wei Kang) — [Scaffold]", value=False)
        
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
                    img_bgr = preprocess_image(img_bgr)
                    if use_morph:
                        pred_m, conf_m, vis_m, met_m, steps_m = analyze_ripeness_by_morphology(img_bgr)
                        results['morph'] = {'pred': pred_m, 'conf': conf_m, 'vis': vis_m, 'metrics': met_m, 'steps': steps_m, 'author': 'Cham Herman', 'name': 'Morphological Blemish', 'status': 'completed'}
                        
                    if use_color:
                        pred_c, conf_c, vis_c, met_c, steps_c = analyze_ripeness_by_color(img_bgr)
                        results['color'] = {'pred': pred_c, 'conf': conf_c, 'vis': vis_c, 'metrics': met_c, 'steps': steps_c, 'author': 'Lum Siew Feng', 'name': 'Color-Space Analysis', 'status': 'completed'}
                        
                    if use_texture:
                        pred_t, conf_t, vis_t, met_t, steps_t = analyze_ripeness_by_texture(img_bgr)
                        results['texture'] = {'pred': pred_t, 'conf': conf_t, 'vis': vis_t, 'metrics': met_t, 'steps': steps_t, 'author': 'Wong Kai Bin', 'name': 'Texture & Surface GLCM', 'status': 'scaffold'}
                        
                    if use_geom:
                        pred_g, conf_g, vis_g, met_g, steps_g = analyze_ripeness_by_geometry(img_bgr)
                        results['geom'] = {'pred': pred_g, 'conf': conf_g, 'vis': vis_g, 'metrics': met_g, 'steps': steps_g, 'author': 'Yeow Wei Kang', 'name': 'Edge & Shape Geometry', 'status': 'scaffold'}
                        
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
                        'preprocessed_bgr': img_bgr
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
            st.markdown("<div style='font-size: 0.85rem; opacity: 0.75; margin-bottom: 12px;'>Expand each tab to inspect the classical image processing operations applied to the mango:</div>", unsafe_allow_html=True)
            
            for k, item in res_dict.items():
                badge_text = "Verified Module" if item.get('status') == 'completed' else "Scaffold Pipeline"
                with st.expander(f"Pipeline: {item['name']} (By {item['author']} — {badge_text})", expanded=(k=='morph')):
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
                        # For up to 5 steps, display in 1 row; for 6+ steps, use fixed 4-column grid so images remain uniform in size
                        cols_per_row = 5 if len(step_items) > 5 else len(step_items)
                        for chunk_start in range(0, len(step_items), cols_per_row):
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
                    row['Primary Physical Metric'] = f"Blemish Area: {item['metrics'].get('blemish_area_ratio', 0):.2f}% (Count: {item['metrics'].get('n_blemishes', 0)})"
                elif k == 'color':
                    all_p = item['metrics'].get('predictions_per_color_space', {})
                    preds_str = ", ".join([f"{cs}:{p}" for cs, p in all_p.items()]) if all_p else ""
                    row['Primary Physical Metric'] = f"Benchmark Acc: {item['metrics'].get('accuracy_benchmark', '97.22%')} | Multi-Model: [{preds_str}]"
                elif k == 'texture':
                    row['Primary Physical Metric'] = f"GLCM Contrast: {item['metrics'].get('glcm_contrast', 0):.1f} | Roughness: {item['metrics'].get('surface_roughness', 0):.1f}"
                elif k == 'geom':
                    row['Primary Physical Metric'] = f"Edge Density: {item['metrics'].get('edge_density_pct', 0):.2f}% | Aspect Ratio: {item['metrics'].get('bounding_aspect_ratio', 0):.2f}"
                    
                table_rows.append(row)
                
            df_metrics = pd.DataFrame(table_rows)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # PDF Report Export Option (Extra Effort)
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
                        'total': 1,
                        'unripe': 1 if pack['consensus']=='unripe' else 0,
                        'fully_ripe': 1 if pack['consensus']=='fully_ripe' else 0,
                        'overripe': 1 if pack['consensus']=='overripe' else 0,
                        'dominant': pack['consensus']
                    }
                    pdf_file = generate_pdf_report(rep_results, summary_stats)
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            label="Download Generated PDF Report",
                            data=f.read(),
                            file_name=f"Inspection_Report_{pack['filename']}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

# -----------------------------------------------------------------------------
# PAGE 2: BULK BATCH ASSESSMENT (CONVEYOR SIMULATION) - EXTRA EFFORT
# -----------------------------------------------------------------------------
elif selected_page.startswith("Bulk"):
    st.markdown(f"<div class='main-title'>{SVG_ICONS['conveyor']} Bulk Batch Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Industrial-grade bulk ingestion system for real-time batch grading across conveyor streams (Extra Efforts).</div>", unsafe_allow_html=True)
    
    # State key for clearing batch images
    if 'batch_uploader_key' not in st.session_state:
        st.session_state['batch_uploader_key'] = 0
        
    uploaded_files = st.file_uploader(
        "Select Multiple Mango Images for Batch Evaluation",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"batch_uploader_{st.session_state['batch_uploader_key']}"
    )
    
    if uploaded_files:
        st.markdown(f"<div class='glass-card'>Loaded <b>{len(uploaded_files)}</b> images for batch processing.</div>", unsafe_allow_html=True)
        
        btn_c1, btn_c2 = st.columns([2, 1])
        with btn_c1:
            run_batch = st.button("Run Batch Analysis on Ingested Stream", type="primary", use_container_width=True)
        with btn_c2:
            clear_batch = st.button("Delete All Images / Clear Batch", use_container_width=True)
            
        if clear_batch:
            st.session_state['batch_uploader_key'] += 1
            if 'batch_df' in st.session_state:
                del st.session_state['batch_df']
            if 'batch_results' in st.session_state:
                del st.session_state['batch_results']
            st.rerun()
            
        if run_batch or 'batch_df' in st.session_state:
            if run_batch:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                t_batch_start = time.time()
                
                for i, f in enumerate(uploaded_files):
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Execute verified modules (Herman & Siew Feng) + scaffolds
                    pred_m, conf_m, _, met_m, _ = analyze_ripeness_by_morphology(bgr)
                    pred_c, conf_c, _, _, _ = analyze_ripeness_by_color(bgr)
                    pred_t, conf_t, _, _, _ = analyze_ripeness_by_texture(bgr)
                    pred_g, conf_g, _, _, _ = analyze_ripeness_by_geometry(bgr)
                    
                    votes = [pred_m, pred_c, pred_t, pred_g]
                    consensus = max(set(votes), key=votes.count)
                    
                    batch_results.append({
                        'filename': f.name,
                        'morph_pred': pred_m,
                        'morph_conf': conf_m,
                        'color_pred': pred_c,
                        'color_conf': conf_c,
                        'texture_pred': pred_t,
                        'texture_conf': conf_t,
                        'geom_pred': pred_g,
                        'geom_conf': conf_g,
                        'final_pred': consensus,
                        'blemish_ratio': met_m.get('blemish_area_ratio', 0)
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    status_text.text(f"Evaluated image {i+1} of {len(uploaded_files)}: {f.name}")
                    
                elapsed_batch = time.time() - t_batch_start
                avg_lat = (elapsed_batch / len(uploaded_files)) * 1000.0
                
                status_text.success(f"Ingested {len(uploaded_files)} images in {elapsed_batch:.2f}s ({avg_lat:.1f} ms/image).")
                
                st.session_state['batch_df'] = pd.DataFrame(batch_results)
                st.session_state['batch_results'] = batch_results
                
            if 'batch_df' in st.session_state:
                df_batch = st.session_state['batch_df']
                batch_results = st.session_state['batch_results']
                
                # Summary Metrics Row
                counts = df_batch['final_pred'].value_counts()
                c_unripe = counts.get('unripe', 0)
                c_ripe = counts.get('fully_ripe', 0)
                c_overripe = counts.get('overripe', 0)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='glass-card'><div class='metric-label'>Total Ingested</div><div class='metric-val'>{len(df_batch)}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='glass-card'><div class='metric-label'>Unripe (Stage 0)</div><div class='metric-val' style='color:#16a34a;'>{c_unripe}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='glass-card'><div class='metric-label'>Fully Ripe (Stage 3)</div><div class='metric-val' style='color:#d97706;'>{c_ripe}</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='glass-card'><div class='metric-label'>Overripe (Defective)</div><div class='metric-val' style='color:#dc2626;'>{c_overripe}</div></div>", unsafe_allow_html=True)
                
                # Batch Distribution Chart
                fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='none')
                ax.set_facecolor('none')
                sns.countplot(data=df_batch, x='final_pred', order=['unripe', 'fully_ripe', 'overripe'], palette=['#16a34a', '#f59e0b', '#dc2626'], ax=ax)
                ax.set_title("Batch Maturity Distribution", fontsize=11, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                
                # Data Table
                st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} Batch Assessment Records</div>", unsafe_allow_html=True)
                st.dataframe(df_batch[['filename', 'morph_pred', 'color_pred', 'texture_pred', 'geom_pred', 'final_pred', 'blemish_ratio']], use_container_width=True, hide_index=True)
                
                # Batch PDF Download
                dominant = df_batch['final_pred'].mode()[0] if not df_batch.empty else 'N/A'
                summary_stats = {
                    'total': len(df_batch),
                    'unripe': c_unripe,
                    'fully_ripe': c_ripe,
                    'overripe': c_overripe,
                    'dominant': dominant
                }
                batch_pdf = generate_pdf_report(batch_results, summary_stats)
                with open(batch_pdf, "rb") as f:
                    st.download_button(
                        label="Download Complete Batch Inspection PDF Report",
                        data=f.read(),
                        file_name="Batch_Inspection_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

# -----------------------------------------------------------------------------
# PAGE 3: SYSTEM ANALYTICS & MODE A BENCHMARKS
# -----------------------------------------------------------------------------
elif selected_page.startswith("System"):
    st.markdown(f"<div class='main-title'>{SVG_ICONS['analytics']} System Analytics & Comparative Benchmark</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Empirical evaluation results and developmental status across team modules.</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} Mode A Table 2.1: Comparative Benchmark Across Image Processing Modules</div>", unsafe_allow_html=True)
    
    benchmark_data = [
        {
            'Algorithm / Module': '1. Morphological Blemish Analysis (Cham Herman)',
            'Development Status': 'Completed & Evaluated',
            'Core Formulation': 'Multi-Scale Beucher Gradient & Black-Hat Residual Fusion',
            'Test Accuracy (%)': '93.06%',
            'Macro F1 (%)': '93.10%',
            'Latency (ms/img)': '29.26 ms'
        },
        {
            'Algorithm / Module': '2. Color-Space Analysis (Lum Siew Feng)',
            'Development Status': 'Completed & Evaluated',
            'Core Formulation': 'Multi-Color Space Chrominance & Support Vector Classification (RGB/HSV/LAB/YCbCr/HLS)',
            'Test Accuracy (%)': '97.22%',
            'Macro F1 (%)': '97.22%',
            'Latency (ms/img)': '12.45 ms'
        },
        {
            'Algorithm / Module': '3. Texture & Surface Analysis (Wong Kai Bin)',
            'Development Status': 'Pending Notebook (Scaffold)',
            'Core Formulation': 'GLCM Contrast & Sobel Roughness (Baseline Prototype)',
            'Test Accuracy (%)': 'Pending',
            'Macro F1 (%)': 'Pending',
            'Latency (ms/img)': '~24.80 ms'
        },
        {
            'Algorithm / Module': '4. Edge & Shape Deformity (Yeow Wei Kang)',
            'Development Status': 'Pending Notebook (Scaffold)',
            'Core Formulation': 'Canny Edge Density & Contour Circularity (Baseline Prototype)',
            'Test Accuracy (%)': 'Pending',
            'Macro F1 (%)': 'Pending',
            'Latency (ms/img)': '~15.60 ms'
        }
    ]
    
    df_bm = pd.DataFrame(benchmark_data)
    st.dataframe(df_bm, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-header'>{SVG_ICONS['analytics']} Verified Modules Performance Visualisation</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    verified_df = pd.DataFrame([
        {'Module': 'Morphology (Herman)', 'Test Accuracy (%)': 93.06, 'Latency (ms)': 29.26},
        {'Module': 'Color-Space (Siew Feng)', 'Test Accuracy (%)': 97.22, 'Latency (ms)': 12.45}
    ])
    
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='none')
        ax.set_facecolor('none')
        sns.barplot(data=verified_df, x='Module', y='Test Accuracy (%)', palette=['#3b82f6', '#f59e0b'], ax=ax)
        ax.set_title("Test Accuracy Comparison (Verified Modules)", fontsize=10, fontweight='bold')
        ax.set_ylim(70, 100)
        ax.axhline(85, color='#ef4444', linestyle='--', label='Target Accuracy (85%)')
        ax.legend(facecolor='none', edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='none')
        ax.set_facecolor('none')
        sns.barplot(data=verified_df, x='Module', y='Latency (ms)', palette=['#10b981', '#6366f1'], ax=ax)
        ax.set_title("Processing Latency per Image (ms)", fontsize=10, fontweight='bold')
        ax.axhline(200, color='#ef4444', linestyle='--', label='Max Target Latency (200 ms)')
        ax.legend(facecolor='none', edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
    st.markdown(f"<div class='section-header'>{SVG_ICONS['table']} SMART Objectives Verification Matrix</div>", unsafe_allow_html=True)
    smart_data = [
        {
            'SMART Objective': 'Objective 1: Multi-Algorithm Suite',
            'Target Criterion': 'Implement 4 distinct classical computer vision algorithms',
            'Current Measured Status': '2 Completed (Herman & Siew Feng) + 2 Scaffolds (Kai Bin & Wei Kang)',
            'Fulfillment': 'In Progress (50% Finalized)'
        },
        {
            'SMART Objective': 'Objective 2: Classification Accuracy',
            'Target Criterion': 'Achieve minimum >= 85% classification accuracy',
            'Current Measured Status': '93.06% (Herman Morphology) | 97.22% (Siew Feng Color)',
            'Fulfillment': 'Target Exceeded'
        },
        {
            'SMART Objective': 'Objective 3: Operational Latency',
            'Target Criterion': 'Execute with per-image latency < 200 ms',
            'Current Measured Status': '29.26 ms (Herman) | 12.45 ms (Siew Feng)',
            'Fulfillment': 'Target Exceeded'
        }
    ]
    st.dataframe(pd.DataFrame(smart_data), use_container_width=True, hide_index=True)
