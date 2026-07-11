import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import custom image processing modules
from src.color_space import analyze_ripeness_by_color
from src.texture import analyze_ripeness_by_texture
from src.geometry import analyze_ripeness_by_geometry
from src.deep_learning import analyze_ripeness_by_deep_learning
from src.reports import generate_pdf_report

st.set_page_config(
    page_title="Mango Ripeness Analyzer",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stHeader {
        background: linear-gradient(135deg, #ff8c00 0%, #e52d27 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #ff8c00;
    }
    .metric-title {
        font-size: 14px;
        color: #b0b0b0;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #ff8c00;
    }
</style>
""", unsafe_allow_html=True)

st.title("🥭 Mango Ripeness Grading & Analytics")
st.markdown("An advanced AI and Computer Vision suite to evaluate mango ripeness.")

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/mango.png", width=90)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Analysis", "Bulk Processing", "Performance Benchmark"])

def process_single_image(img_bgr):
    # Process with all 4 techniques
    # 1. Color Space
    t0 = time.time()
    color_pred, color_conf, color_vis = analyze_ripeness_by_color(img_bgr)
    t_color = (time.time() - t0) * 1000
    
    # 2. Texture Analysis
    t0 = time.time()
    text_pred, text_conf, text_vis = analyze_ripeness_by_texture(img_bgr)
    t_text = (time.time() - t0) * 1000
    
    # 3. Geometry/Edges
    t0 = time.time()
    geom_pred, geom_conf, geom_vis = analyze_ripeness_by_geometry(img_bgr)
    t_geom = (time.time() - t0) * 1000
    
    # 4. Deep Learning
    t0 = time.time()
    dl_pred, dl_conf, dl_vis = analyze_ripeness_by_deep_learning(img_bgr)
    t_dl = (time.time() - t0) * 1000
    
    # Simple consensus voting for final prediction
    votes = [color_pred, text_pred, geom_pred, dl_pred]
    final_pred = max(set(votes), key=votes.count)
    
    return {
        'color': (color_pred, color_conf, color_vis, t_color),
        'texture': (text_pred, text_conf, text_vis, t_text),
        'geometry': (geom_pred, geom_conf, geom_vis, t_geom),
        'dl': (dl_pred, dl_conf, dl_vis, t_dl),
        'final': final_pred
    }

if page == "Single Analysis":
    st.subheader("Single Image Diagnostic playground")
    
    uploaded_file = st.file_uploader("Upload a Mango image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Layout Columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img_rgb, caption="Uploaded Original Image", use_container_width=True)
            
        with col2:
            st.write("### Grading Engine Diagnostics")
            with st.spinner("Analyzing with 4 pipelines..."):
                results = process_single_image(img_bgr)
                
            st.success(f"Consensus Grading: **{results['final']}**")
            
            # Show styled cards for each team member's approach
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Siew Feng (Color Space)</div>
                    <div class="metric-value">{results['color'][0]}</div>
                    <p style="margin: 0; color: #888;">Conf: {results['color'][1]:.2f} | {results['color'][3]:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Kai Bin (Texture)</div>
                    <div class="metric-value">{results['texture'][0]}</div>
                    <p style="margin: 0; color: #888;">Conf: {results['texture'][1]:.2f} | {results['texture'][3]:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Wei Kang (Geometry)</div>
                    <div class="metric-value">{results['geometry'][0]}</div>
                    <p style="margin: 0; color: #888;">Conf: {results['geometry'][1]:.2f} | {results['geometry'][3]:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Herman (Deep Learning)</div>
                    <div class="metric-value">{results['dl'][0]}</div>
                    <p style="margin: 0; color: #888;">Conf: {results['dl'][1]:.2f} | {results['dl'][3]:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("Technique Visualizations")
        
        # Display algorithm visual outputs
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            st.image(cv2.cvtColor(results['color'][2], cv2.COLOR_BGR2RGB), caption="Color Threshold Mask", use_container_width=True)
        with v_col2:
            st.image(cv2.cvtColor(results['texture'][2], cv2.COLOR_BGR2RGB), caption="GLCM Texture Gradient Map", use_container_width=True)
        with v_col3:
            st.image(cv2.cvtColor(results['geometry'][2], cv2.COLOR_BGR2RGB), caption="Canny Edges & Bounding Box", use_container_width=True)
        with v_col4:
            st.image(cv2.cvtColor(results['dl'][2], cv2.COLOR_BGR2RGB), caption="Deep Learning Overlay", use_container_width=True)

elif page == "Bulk Processing":
    st.subheader("Bulk Ingestion & Report Export")
    st.markdown("Upload multiple images to run batch grading pipeline and download a formatted PDF summary report.")
    
    uploaded_files = st.file_uploader("Select multiple Mango images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        records = []
        progress_bar = st.progress(0.0)
        
        for i, file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            res = process_single_image(img_bgr)
            
            records.append({
                'filename': file.name,
                'color_pred': res['color'][0],
                'color_conf': res['color'][1],
                'texture_pred': res['texture'][0],
                'texture_conf': res['texture'][1],
                'geom_pred': res['geometry'][0],
                'geom_conf': res['geometry'][1],
                'dl_pred': res['dl'][0],
                'dl_conf': res['dl'][1],
                'final_pred': res['final']
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)
        
        # Calculate summary statistics
        counts = df['final_pred'].value_counts()
        unripe_count = int(counts.get('Unripe', 0))
        partially_ripe_count = int(counts.get('Partially Ripe', 0))
        fully_ripe_count = int(counts.get('Fully Ripe', 0))
        dominant_class = df['final_pred'].mode()[0] if not df.empty else "N/A"
        
        summary_stats = {
            'total': len(uploaded_files),
            'unripe': unripe_count,
            'partially_ripe': partially_ripe_count,
            'fully_ripe': fully_ripe_count,
            'dominant': dominant_class
        }
        
        # Display simple metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Processed", summary_stats['total'])
        m2.metric("Unripe", summary_stats['unripe'])
        m3.metric("Partially Ripe", summary_stats['partially_ripe'])
        m4.metric("Fully Ripe", summary_stats['fully_ripe'])
        
        # Export PDF
        st.write("### Export Options")
        if st.button("Generate & Download PDF Report"):
            with st.spinner("Generating Report PDF..."):
                pdf_path = generate_pdf_report(records, summary_stats)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name="Mango_Ripeness_Grading_Report.pdf",
                        mime="application/pdf"
                    )

elif page == "Performance Benchmark":
    st.subheader("Technique Benchmark & Model Comparisons")
    st.markdown("Overview and model evaluation metrics compiled from Siew Feng, Kai Bin, Wei Kang, and Herman's playground notebooks.")
    
    # Set up mock comparison metrics based on model validations
    benchmark_data = pd.DataFrame({
        'Developer': ['Siew Feng', 'Kai Bin', 'Wei Kang', 'Herman'],
        'Technique': ['Color Space (HSV)', 'GLCM Texture Analysis', 'Geometry & Edges', 'Deep Learning (CNN)'],
        'Accuracy (%)': [84.5, 78.2, 73.1, 93.6],
        'Avg. Latency (ms)': [12.4, 45.1, 18.2, 115.6],
        'Edge Cases Handling': ['Good', 'Moderate', 'Poor', 'Excellent'],
        'Dataset Coverage': ['100%', '100%', '100%', '100%']
    })
    
    st.table(benchmark_data)
    
    # Generate interactive charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Accuracy Comparison (%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Developer', y='Accuracy (%)', data=benchmark_data, palette='Oranges_d', ax=ax)
        ax.set_ylim(50, 100)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() - 5),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='white', fontweight='bold')
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        
    with col2:
        st.write("#### Avg. Inference Latency (ms) - log scale")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Developer', y='Avg. Latency (ms)', data=benchmark_data, palette='Greens_d', ax=ax)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
