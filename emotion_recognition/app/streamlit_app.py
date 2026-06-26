import time
import cv2
import streamlit as st
# import numpy as np TODO remove

from emotion_recognition.interface.pipeline import FERPipeline
# from emotion_recognition.params import EMOTIONS_CLASSES TODO remove
from app.video_capture import VideoStream



# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Real Time Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
# All stateful objects live in session_state so they survive Streamlit reruns.
# Critically: FERPipeline must persist across frames so the temporal smoother
# retains its rolling window — recreating it every frame would silently reset
# smoothing to nothing.

if "pipeline" not in st.session_state:
    st.session_state.pipeline = FERPipeline(
        draw=False   # Handle drawing in this file, not inside the pipeline
    )

if "stream" not in st.session_state:
    st.session_state.stream = None

if "running" not in st.session_state:
    st.session_state.running = False

if "fps_history" not in st.session_state:
    st.session_state.fps_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("FER Demo")
    st.markdown("Real-time facial emotion recognition using a custom CNN architecture and Google's MediaPipe for the face detection.")
    st.divider()

    st.subheader("Controls")
    camera_index = st.selectbox("Camera", options=[0, 1, 2], index=0,
                                 help="Try 1 or 2 if your primary webcam is not index 0.")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        help="Predictions below this confidence are flagged as uncertain."
    )
    show_all_scores = st.toggle("Show all emotion scores", value=True)
    show_timing = st.toggle("Show timing breakdown", value=False)

    st.divider()
    st.subheader("Model")
    st.caption("Custom CNN · ONNX INT8 · MediaPipe face detection · Temporal smoothing (EMA α=0.3)")

# ── Layout ────────────────────────────────────────────────────────────────────

st.title("Facial Emotion Recognition")

col_feed, col_stats = st.columns([2, 1], gap="large")

with col_feed:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("▶ Start", width='stretch', type="primary",       # TODO use_container_width=True
                               disabled=st.session_state.running)
    with btn_col2:
        stop_btn = st.button("⏹ Stop", width='stretch',                         # TODO use_container_width=True
                              disabled=not st.session_state.running)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with col_stats:
    st.subheader("Detected emotions")
    emotion_placeholder = st.empty()
    st.divider()
    fps_placeholder = st.empty()
    timing_placeholder = st.empty()

# ── Button handlers ───────────────────────────────────────────────────────────

if start_btn:
    st.session_state.running = True
    st.session_state.stream = VideoStream(src=camera_index)
    st.rerun()

if stop_btn:
    st.session_state.running = False
    if st.session_state.stream is not None:
        st.session_state.stream.stop()
        st.session_state.stream = None
    frame_placeholder.empty()
    status_placeholder.info("Stream stopped.")
    st.rerun()

# ── Idle state ────────────────────────────────────────────────────────────────

if not st.session_state.running:
    frame_placeholder.markdown(
        """
        <div style="
            height: 360px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1.5px dashed #888;
            border-radius: 12px;
            color: #888;
            font-size: 1.1rem;
        ">
            Press <strong>&nbsp;▶ Start&nbsp;</strong> to begin
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Live loop ─────────────────────────────────────────────────────────────────

stream: VideoStream = st.session_state.stream
pipeline: FERPipeline = st.session_state.pipeline

if not stream.is_opened():
    st.error("Could not open camera. Try a different camera index in the sidebar.")
    st.session_state.running = False
    st.stop()

fps_tracker = []   # rolling timestamps for FPS calculation

while st.session_state.running:
    # ── Capture ──────────────────────────────────────────────────────────────
    t0 = time.time()
    ret, frame = stream.read()
    t1 = time.time()

    if not ret or frame is None:
        status_placeholder.warning("Frame not available — retrying...")
        time.sleep(0.05)
        continue

    # ── Inference ─────────────────────────────────────────────────────────────
    _, results = pipeline.pipeline_flow(frame)
    t2 = time.time()

    # ── Draw bounding boxes with current frame coords ─────────────────────────
    img = frame.copy()
    for face_id, data in results.items():
        bbox = data["bounding_box"]
        pred_class = data["predicted_class"]
        pred_prob = data["predicted_prob"]
        is_confident = pred_prob >= confidence_threshold

        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        thickness = 1
        length = 30

        color = (255, 255, 255) if is_confident else (0, 200, 220)   # green / yellow
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color, thickness
        )

        # Top left corner
        cv2.line(img, (x, y), (x+length, y), (255, 0, 255), thickness*2)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness*2)

        # Top right corner
        cv2.line(img, (x1, y), (x1-length, y), (255, 0, 255), thickness*2)
        cv2.line(img, (x1, y), (x1, y+length), (255, 0, 255), thickness*2)

        # Bottom left corner
        cv2.line(img, (x, y1), (x, y1-length), (255, 0, 255), thickness*2)
        cv2.line(img, (x, y1), (x+length, y1), (255, 0, 255), thickness*2)

        # Bottom right corner
        cv2.line(img, (x1, y1), (x1, y1-length), (255, 0, 255), thickness*2)
        cv2.line(img, (x1, y1), (x1-length, y1), (255, 0, 255), thickness*2)


        label = f"{pred_class.capitalize()} ({pred_prob:.0%})" if is_confident else f"? {pred_class} {pred_prob:.0%}"
        cv2.putText(img,
                    label,
                    (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)

    t3 = time.time()

    # ── Display frame ─────────────────────────────────────────────────────────
    frame_placeholder.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        width='stretch'                                                         # TODO use_container_width=True
    )
    t4 = time.time()

    # ── FPS ───────────────────────────────────────────────────────────────────
    now = time.time()
    fps_tracker.append(now)
    fps_tracker = [t for t in fps_tracker if now - t <= 1.0]   # keep last 1 second
    current_fps = len(fps_tracker)

    # ── Stats panel ───────────────────────────────────────────────────────────
    if results:
        with emotion_placeholder.container():
            for face_id, data in results.items():
                pred_class = data["predicted_class"]
                pred_prob = data["predicted_prob"]
                is_confident = pred_prob >= confidence_threshold

                if is_confident:
                    st.success(f"**{pred_class}** — {pred_prob:.0%} confidence")
                else:
                    st.warning(f"⚠️ Uncertain — closest: **{pred_class}** ({pred_prob:.0%})")

                if show_all_scores:
                    scores = data["proba_by_class"]

                    # Sort by score descending for readability
                    sorted_scores = dict(
                        sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    st.bar_chart(sorted_scores, height=180)

    else:
        emotion_placeholder.error(body="No face detected",
                                  icon="❌")

    with fps_placeholder.container():
        st.metric("FPS", current_fps)

    if show_timing:
        with timing_placeholder.container():
            st.caption("Timing breakdown (ms)")
            st.markdown(
               f"""
                |   Stage    |             Time            |
                |------------|-----------------------------|
                | Capture    | `{(t1 - t0) * 1000:.1f} ms` |
                | Inference  | `{(t2 - t1) * 1000:.1f} ms` |
                | Draw       | `{(t3 - t2) * 1000:.1f} ms` |
                | Display    | `{(t4 - t3) * 1000:.1f} ms` |
                """
            )
