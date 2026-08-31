"""Live Camera Inspection (Extra Effort): real-time per-frame ripeness analysis.

Assignment spec — Video Processing: "Enable ingestion of video streams as input
data and execute real-time algorithmic analysis across individual frames." The
webcam is the video stream: streamlit-webrtc delivers frames to a worker-thread
callback, each frame is analysed with the user's selected algorithm from the 4
teammate pipelines, and the annotated frame is returned to the browser continuously.

Background handling: live camera frames contain real backgrounds (hands,
desks), so every frame is first segmented with the background-agnostic
morphological masker before the analyzers' black-background threshold
assumptions apply.
"""
import collections
import threading
import time

import cv2
import numpy as np

from src.morphology import analyze_ripeness_by_morphology
from src.color_space import analyze_ripeness_by_color
from src.texture import analyze_ripeness_by_texture
from src.geometry import analyze_ripeness_by_geometry
from src.preprocessing import segment_fruit_mask, remove_background, remove_noise, enhance_contrast
from src.hardware import get_hardware_info, init_hardware_acceleration

# Auto-initialize hardware acceleration (GPU if available, fallback to CPU)
_IS_GPU_ENABLED, _ACTIVE_DEVICE_NAME = init_hardware_acceleration(enable_gpu=True)

CLASS_COLORS_RGB = {
    'unripe': (44, 160, 44),
    'fully_ripe': (255, 127, 14),
    'overripe': (214, 39, 40),
}

PREPROCESSING_METADATA = {
    'kmeans': {
        'name': 'Standard K-Means & Convex Hull (Default)',
        'tag': 'K-Means + Hull',
    },
    'morphology': {
        'name': 'Background-Agnostic Morphological Masking',
        'tag': 'Morph Mask',
    }
}

ALGORITHM_METADATA = {
    'morphology': {
        'name': 'Morphology (Cham Herman)',
        'tag': 'MRMF Morphology',
        'author': 'Cham Herman'
    },
    'color': {
        'name': 'Color-Space (Lum Siew Feng)',
        'tag': 'Color-Space (LAB)',
        'author': 'Lum Siew Feng'
    },
    'texture': {
        'name': 'Texture Analysis (Wong Kai Bin)',
        'tag': 'GLCM/LBP Texture',
        'author': 'Wong Kai Bin'
    },
    'geometry': {
        'name': 'Edge & Geometry (Yeow Wei Kang)',
        'tag': 'Scharr Edge/Geom',
        'author': 'Yeow Wei Kang'
    },
}


def letterbox_640(image_bgr, size=640):
    """Letterbox an arbitrary frame onto a black square canvas (the same
    standardisation as the cleaned_data training set)."""
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def analyze_frame(image_bgr, algorithm="morphology", preprocessing="kmeans", model_path=None):
    """Analyse one camera frame using the chosen preprocessing backend and student algorithm:
    Letterbox -> Selected Preprocessing (K-Means Siew Feng / Morphology Herman) -> Model Inference ->
    HUD verdict banner + algorithm-specific visual overlay.

    Args:
        image_bgr (np.ndarray): Input frame from camera or video stream.
        algorithm (str): One of ['morphology', 'color', 'texture', 'geometry'].
        preprocessing (str): One of ['kmeans', 'morphology']. Default is 'kmeans' (Siew Feng).
        model_path (str, optional): Custom path for model if applicable.

    Returns:
        dict with keys: 'prediction', 'confidence', 'latency_ms',
                        'annotated_rgb', 'tierb_mask', 'metrics', 'algorithm',
                        'preprocessing', 'preprocessing_name'
    """
    t_start = time.perf_counter()
    frame = letterbox_640(image_bgr)
    
    # 1. Execute Selected Preprocessing Engine
    prep_key = str(preprocessing).lower()
    if 'morph' in prep_key:
        prep_type = 'morphology'
        fruit_mask = segment_fruit_mask(frame)
        frame_bg = cv2.bitwise_and(frame, frame, mask=fruit_mask)
    else:
        prep_type = 'kmeans'
        denoised = remove_noise(frame)
        contrasted = enhance_contrast(denoised)
        frame_bg, fruit_mask = remove_background(contrasted)
        if fruit_mask is not None and fruit_mask.max() == 1:
            fruit_mask = (fruit_mask * 255).astype(np.uint8)

    # 2. Execute Selected Ripeness Grading Algorithm
    alg_key = str(algorithm).lower()
    if 'morph' in alg_key:
        alg_type = 'morphology'
        if model_path:
            pred, conf, blended, metrics, _ = analyze_ripeness_by_morphology(frame_bg, model_path)
        else:
            pred, conf, blended, metrics, _ = analyze_ripeness_by_morphology(frame_bg)
        extra_hud = f"Blemish: {metrics.get('blemish_area_ratio', 0.0):.1f}% [{metrics.get('severity_grade', '-')}]"
    elif 'color' in alg_key:
        alg_type = 'color'
        pred, conf, blended, metrics, _ = analyze_ripeness_by_color(frame_bg, primary_space="LAB")
        extra_hud = f"H:{metrics.get('mean_hue', 0):.0f} S:{metrics.get('mean_saturation', 0):.0f} V:{metrics.get('mean_value', 0):.0f}"
    elif 'text' in alg_key:
        alg_type = 'texture'
        pred, conf, blended, metrics, _ = analyze_ripeness_by_texture(frame_bg)
        extra_hud = f"Cont:{metrics.get('glcm_contrast', 0):.1f} | LBP Ent:{metrics.get('lbp_entropy', 0):.2f}"
    elif 'geom' in alg_key or 'edge' in alg_key:
        alg_type = 'geometry'
        pred, conf, blended, metrics, _ = analyze_ripeness_by_geometry(frame_bg)
        extra_hud = f"Edge Dens:{metrics.get('scharr_density', 0)*100:.1f}% | AR:{metrics.get('aspect_ratio', 0):.2f}"
    else:
        alg_type = 'morphology'
        pred, conf, blended, metrics, _ = analyze_ripeness_by_morphology(frame_bg)
        extra_hud = f"Blemish: {metrics.get('blemish_area_ratio', 0.0):.1f}% [{metrics.get('severity_grade', '-')}]"

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    meta = ALGORITHM_METADATA.get(alg_type, ALGORITHM_METADATA['morphology'])
    prep_meta = PREPROCESSING_METADATA.get(prep_type, PREPROCESSING_METADATA['kmeans'])
    tag = meta['tag']
    prep_tag = prep_meta['tag']

    # Compose top HUD banner overlay (2-line structured layout to prevent text overlap)
    annotated = blended.copy() if blended is not None else cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
    color = CLASS_COLORS_RGB.get(pred, (255, 255, 255))

    banner_height = 54
    banner = np.zeros((banner_height, annotated.shape[1], 3), dtype=np.uint8)
    banner[:] = (18, 22, 28)  # Sleek dark background

    # Line 1 (y=24): Algorithm/Preprocessing Tag (Left) & Prominent Ripeness Verdict (Right)
    tag_text = f"[{tag} | {prep_tag}]"
    verdict_text = f"{pred.upper()} ({conf:.1f}%)"
    cv2.putText(banner, tag_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 225, 240), 1, cv2.LINE_AA)
    
    (v_w, _), _ = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    verdict_x = max(annotated.shape[1] - v_w - 12, 320)
    cv2.putText(banner, verdict_text, (verdict_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    # Line 2 (y=44): Secondary Algorithm Metrics (Left) & Measured Latency (Right)
    cv2.putText(banner, extra_hud, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 185, 190), 1, cv2.LINE_AA)
    
    lat_text = f"{total_latency_ms:.0f} ms"
    (l_w, _), _ = cv2.getTextSize(lat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    lat_x = annotated.shape[1] - l_w - 12
    cv2.putText(banner, lat_text, (lat_x, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 155, 160), 1, cv2.LINE_AA)

    # Accent bottom line on banner
    cv2.line(banner, (0, banner_height - 1), (annotated.shape[1], banner_height - 1), color, 2)

    annotated = np.vstack([banner, annotated])

    hw = get_hardware_info()
    return {
        'prediction': pred,
        'confidence': conf,
        'severity_grade': metrics.get('severity_grade', '-'),
        'needs_review': metrics.get('needs_review', False),
        'defect_percentage': metrics.get('defect_percentage', metrics.get('blemish_area_ratio', 0.0)),
        'latency_ms': total_latency_ms,
        'annotated_rgb': annotated,
        'tierb_mask': fruit_mask,
        'metrics': metrics,
        'algorithm': alg_type,
        'algorithm_name': meta['name'],
        'preprocessing': prep_type,
        'preprocessing_name': prep_meta['name'],
        'device': hw['device_type'],
        'device_name': hw['device_name'],
        'backend': hw['backend']
    }


class LiveSessionStats:
    """Thread-safe rolling statistics, rich per-frame telemetry history, and
    runtime configuration for the live camera session.

    The webrtc callback runs in a worker thread while the Streamlit script
    reruns in the main thread, so all access is lock-guarded."""

    def __init__(self, maxlen=1000, fps_window=60, default_algorithm="morphology", default_preprocessing="kmeans"):
        self._lock = threading.Lock()
        self._verdicts = collections.deque(maxlen=maxlen)
        self._latencies = collections.deque(maxlen=maxlen)
        self._frame_times = collections.deque(maxlen=fps_window)
        self._history = collections.deque(maxlen=maxlen)
        self._total = 0
        self._algorithm = default_algorithm
        self._preprocessing = default_preprocessing

    def get_algorithm(self):
        with self._lock:
            return self._algorithm

    def set_algorithm(self, alg_name):
        with self._lock:
            self._algorithm = alg_name

    def get_preprocessing(self):
        with self._lock:
            return self._preprocessing

    def set_preprocessing(self, prep_name):
        with self._lock:
            self._preprocessing = prep_name

    def reset(self):
        """Reset all counters, rolling stats, and telemetry history."""
        with self._lock:
            self._verdicts.clear()
            self._latencies.clear()
            self._frame_times.clear()
            self._history.clear()
            self._total = 0

    def record(self, result_data, latency_ms=None):
        """Record per-frame prediction, latency, compute device, and diagnostic metrics."""
        with self._lock:
            self._total += 1
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            hw = get_hardware_info()

            if isinstance(result_data, dict):
                pred = result_data.get('prediction', 'unknown')
                lat = float(result_data.get('latency_ms', 0.0))
                conf = float(result_data.get('confidence', 0.0))
                alg = result_data.get('algorithm_name', self._algorithm)
                prep = result_data.get('preprocessing_name', self._preprocessing)
                sev = result_data.get('severity_grade', '-')
                def_pct = result_data.get('defect_percentage', 0.0)
                dev_name = result_data.get('device_name', hw['device_name'])
                m = result_data.get('metrics', {})

                entry = {
                    'Frame': self._total,
                    'Timestamp': now_str,
                    'Compute Device': dev_name,
                    'Algorithm': alg,
                    'Preprocessing': prep,
                    'Ripeness': pred.upper(),
                    'Confidence (%)': round(conf, 1),
                    'Latency (ms)': round(lat, 1),
                    'Severity Grade': sev,
                    'Defect (%)': round(float(def_pct), 2) if def_pct != '-' else '-',
                    'Blemish Area (%)': m.get('blemish_area_ratio', '-'),
                    'Max Lesion Ratio (%)': m.get('max_lesion_ratio', '-'),
                    'Mean Hue': m.get('mean_hue', '-'),
                    'Mean Saturation': m.get('mean_saturation', '-'),
                    'Mean Value': m.get('mean_value', '-'),
                    'GLCM Contrast': m.get('glcm_contrast', '-'),
                    'LBP Entropy': m.get('lbp_entropy', '-'),
                    'Edge Density (%)': round(float(m.get('scharr_density', 0.0)) * 100, 2) if 'scharr_density' in m else '-',
                    'Aspect Ratio': m.get('aspect_ratio', '-')
                }
            else:
                pred = str(result_data)
                lat = float(latency_ms or 0.0)
                entry = {
                    'Frame': self._total,
                    'Timestamp': now_str,
                    'Compute Device': hw['device_name'],
                    'Algorithm': self._algorithm,
                    'Preprocessing': self._preprocessing,
                    'Ripeness': pred.upper(),
                    'Confidence (%)': 0.0,
                    'Latency (ms)': round(lat, 1),
                    'Severity Grade': '-',
                    'Defect (%)': '-'
                }

            self._verdicts.append(pred)
            self._latencies.append(lat)
            self._frame_times.append(time.perf_counter())
            self._history.append(entry)

    def snapshot(self):
        with self._lock:
            counts = {c: 0 for c in ('unripe', 'fully_ripe', 'overripe')}
            for v in self._verdicts:
                counts[v] = counts.get(v, 0) + 1
            fps = 0.0
            if len(self._frame_times) >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            return {
                'total_frames': self._total,
                'verdict_counts': counts,
                'measured_fps': fps,
                'avg_latency_ms': avg_lat,
                'algorithm': self._algorithm,
                'preprocessing': self._preprocessing
            }

    def get_history_dataframe(self):
        """Export session telemetry history as a pandas DataFrame."""
        import pandas as pd
        with self._lock:
            if not self._history:
                return pd.DataFrame()
            return pd.DataFrame(list(self._history))


def make_webrtc_callback(algorithm="morphology", preprocessing="kmeans", stats=None, model_path=None):
    """Build a video_frame_callback for streamlit_webrtc.webrtc_streamer.

    Each webcam frame is converted to BGR, processed with the chosen preprocessing
    technique (K-Means Siew Feng / Morphology Herman), analysed with the chosen
    ripeness algorithm, annotated with HUD elements, and returned as an RGB
    VideoFrame for continuous real-time display in the browser."""
    from av import VideoFrame  # provided by the streamlit-webrtc dependency chain

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            active_alg = stats.get_algorithm() if stats is not None else algorithm
            active_prep = stats.get_preprocessing() if stats is not None else preprocessing
            res = analyze_frame(img, algorithm=active_alg, preprocessing=active_prep, model_path=model_path)
        except Exception:
            return frame
        if stats is not None:
            stats.record(res)
        return VideoFrame.from_ndarray(res['annotated_rgb'], format="rgb24")

    return video_frame_callback


