"""Live Camera Inspection (Extra Effort): real-time per-frame ripeness analysis.

Assignment spec — Video Processing: "Enable ingestion of video streams as input
data and execute real-time algorithmic analysis across individual frames." The
webcam is the video stream: streamlit-webrtc delivers frames to a worker-thread
callback, each frame is analysed with the MRMF morphology pipeline,
and the annotated frame is returned to the browser continuously.

Background handling: live camera frames contain real backgrounds (hands,
desks), so every frame is first segmented with the background-agnostic
morphological masker before the analyzers' black-background threshold
assumptions apply.
"""
import collections
import threading

import cv2
import numpy as np

from src.morphology import analyze_ripeness_by_morphology
from src.preprocessing import segment_fruit_mask

CLASS_COLORS_RGB = {
    'unripe': (44, 160, 44),
    'fully_ripe': (255, 127, 14),
    'overripe': (214, 39, 40),
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


def analyze_frame(image_bgr, model_path="output/morphology_based/morphology_model.joblib"):
    """Analyse one camera frame: background segmentation -> MRMF morphology ->
    verdict banner + blemish overlay + lesion callout.

    Returns a result dict with 'annotated_rgb' (RGB, verdict banner on top)."""
    frame = letterbox_640(image_bgr)
    fruit_mask = segment_fruit_mask(frame)
    frame_bg = cv2.bitwise_and(frame, frame, mask=fruit_mask)

    pred, conf, blended, metrics, _ = analyze_ripeness_by_morphology(frame_bg, model_path)

    annotated = blended.copy()
    color = CLASS_COLORS_RGB.get(pred, (255, 255, 255))
    banner = np.zeros((46, annotated.shape[1], 3), dtype=np.uint8)
    label = f"{pred.upper()}  {conf:.1f}%  |  {metrics.get('severity_grade', '-')}"
    cv2.putText(banner, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    annotated = np.vstack([banner, annotated])

    return {
        'prediction': pred,
        'confidence': conf,
        'severity_grade': metrics.get('severity_grade'),
        'needs_review': metrics.get('needs_review', False),
        'defect_percentage': metrics.get('defect_percentage', 0.0),
        'latency_ms': metrics.get('latency_ms', 0.0),
        'annotated_rgb': annotated,
        'tierb_mask': fruit_mask,
    }


class LiveSessionStats:
    """Thread-safe rolling statistics for the live camera session.

    The webrtc callback runs in a worker thread while the Streamlit script
    reruns in the main thread, so all access is lock-guarded."""

    def __init__(self, maxlen=600, fps_window=60):
        self._lock = threading.Lock()
        self._verdicts = collections.deque(maxlen=maxlen)
        self._latencies = collections.deque(maxlen=maxlen)
        self._frame_times = collections.deque(maxlen=fps_window)
        self._total = 0

    def record(self, prediction, latency_ms):
        import time
        with self._lock:
            self._verdicts.append(prediction)
            self._latencies.append(latency_ms)
            self._frame_times.append(time.perf_counter())
            self._total += 1

    def snapshot(self):
        with self._lock:
            counts = {c: 0 for c in ('unripe', 'fully_ripe', 'overripe')}
            for v in self._verdicts:
                counts[v] = counts.get(v, 0) + 1
            n = max(len(self._frame_times), 1)
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
            }


def make_webrtc_callback(model_path="output/morphology_based/morphology_model.joblib",
                         stats=None):
    """Build a video_frame_callback for streamlit_webrtc.webrtc_streamer.

    Each webcam frame is converted to BGR, analysed (background segmentation
    + MRMF morphology), annotated, and returned as an
    RGB VideoFrame for continuous display in the browser."""
    from av import VideoFrame  # provided by the streamlit-webrtc dependency chain

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            res = analyze_frame(img, model_path)
        except Exception:
            return frame
        if stats is not None:
            stats.record(res['prediction'], res['latency_ms'])
        return VideoFrame.from_ndarray(res['annotated_rgb'], format="rgb24")

    return video_frame_callback
