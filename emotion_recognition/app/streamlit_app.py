import time
import cv2
import streamlit as st
import altair as alt
import pandas as pd

from emotion_recognition.interface.pipeline import FERPipeline
from emotion_recognition.app.video_capture import VideoStream


# ------------------------------- Page config -------------------------------- #
st.set_page_config(
    page_title="Real Time Facial Emotion Recognition",
    layout="wide"
)

# ---------------------------- Session state init ---------------------------- #
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


# ---------------------------------- Sidebar --------------------------------- #
with st.sidebar:
    st.title("Facial Emotion Recognition Demo")
    st.markdown(
        """Real-time facial emotion recognition (FER) using a custom Convolutionnal Neural Network architecture.""")
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
    show_fps = st.toggle("Show frame rate", value=False)
    show_nb_faces = st.toggle("Show number of detected faces", value=False)

    st.divider()

    st.subheader("Model")
    st.caption("Custom CNN · ONNX · MediaPipe face detection")


# ----------------------------------- Layout --------------------------------- #
frame_column, stats_column = st.columns([3, 1], gap="small")

with frame_column:
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        start_button = st.button("▶ Start", width='stretch', type="primary",
                               disabled=st.session_state.running)
    with button_col2:
        stop_button = st.button("⏹ Stop", width='stretch',
                              disabled=not st.session_state.running)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    fps_col, nb_face_col = st.columns(2, gap="small")

    with fps_col:
        fps_placeholder = st.empty()

    with nb_face_col:
        nb_faces_placeholder = st.empty()


with stats_column:
    st.subheader("Detected emotions")
    emotion_placeholder = st.empty()
    st.divider()


# ----------------------------- Button handlers ------------------------------ #
if start_button:
    st.session_state.running = True
    st.session_state.stream = VideoStream(camera_index)
    st.rerun()

if stop_button:
    st.session_state.running = False
    if st.session_state.stream is not None:
        st.session_state.stream.stop()
        st.session_state.stream = None
    frame_placeholder.empty()
    status_placeholder.info("Stream stopped.")
    st.rerun()

# ------------------------------ Idle state ---------------------------------- #
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

# ------------------------------- Live loop ---------------------------------- #
stream: VideoStream = st.session_state.stream
pipeline: FERPipeline = st.session_state.pipeline

if not stream.is_opened():
    st.error("Could not open camera. Try a different camera index in the sidebar.")
    st.session_state.running = False
    st.stop()

fps_tracker = [] # rolling timestamps for FPS calculation

while st.session_state.running:

    # ------------------------------ Capture --------------------------------- #
    ret, frame = stream.read()

    if not ret or frame is None:
        status_placeholder.warning("Frame not available — retrying...")
        time.sleep(0.05)
        continue

    # ----------------------------- Inference -------------------------------- #
    _, results = pipeline.pipeline_flow(frame)

    img = frame.copy()
    for face_id, data in results.items():
        bbox = data["bounding_box"]
        pred_class = data["predicted_class"]
        pred_prob = data["predicted_prob"]
        is_confident = pred_prob >= confidence_threshold

        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        thickness = 2
        length = 40

        color = (255, 255, 255) if is_confident else (0, 200, 220)   # white (certain) / yellow (uncertain)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness) # Bounding Box

        corners_colors = (255, 0, 255) if is_confident else (0, 0, 220) # purple (certain) / red (uncertain)

        # Top left corner
        cv2.line(img, (x, y), (x+length, y), corners_colors, thickness*2)
        cv2.line(img, (x, y), (x, y+length), corners_colors, thickness*2)

        # Top right corner
        cv2.line(img, (x1, y), (x1-length, y), corners_colors, thickness*2)
        cv2.line(img, (x1, y), (x1, y+length), corners_colors, thickness*2)

        # Bottom left corner
        cv2.line(img, (x, y1), (x, y1-length), corners_colors, thickness*2)
        cv2.line(img, (x, y1), (x+length, y1), corners_colors, thickness*2)

        # Bottom right corner
        cv2.line(img, (x1, y1), (x1, y1-length), corners_colors, thickness*2)
        cv2.line(img, (x1, y1), (x1-length, y1), corners_colors, thickness*2)


        label = f"Face {face_id+1}: {pred_class} ({pred_prob:.0%})"

        cv2.putText(img,
                    label,
                    (x, max(y - 20, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness) #0.65


    # --------------------------- Display frame ------------------------------ #
    frame_placeholder.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        width='stretch')

    # -------------------------------- FPS ----------------------------------- #
    now = time.time()
    fps_tracker.append(now)
    fps_tracker = [t for t in fps_tracker if now - t <= 1.0] # keep last 1 second
    current_fps = len(fps_tracker)

    # ---------------------------- Stats panel ------------------------------- #
    if results:
        with emotion_placeholder.container():
            for face_id, data in results.items():
                pred_class = data["predicted_class"]
                pred_prob = data["predicted_prob"]
                is_confident = pred_prob >= confidence_threshold

                if is_confident:
                    st.success(f"**Face {face_id+1}: {pred_class}** ({pred_prob:.0%})")
                else:
                    st.warning(f"⚠️ [Uncertain] **Face {face_id+1}: {pred_class}** ({pred_prob:.0%})")

                if show_all_scores:
                    confidences = data["proba_by_class"]

                    # Sort by score descending for readability
                    sorted_scores = dict(
                        sorted(confidences.items(), key=lambda x: x[1], reverse=True)
                    )

                    conf_scores = pd.DataFrame(list(sorted_scores.items()), columns=["Emotion", "Confidence"])

                    # Horizontal bar chart with fixed scale
                    chart = alt.Chart(conf_scores).mark_bar(
                        color="#fe4c4a",
                        width=12 # Bar width
                    ).encode(
                        y=alt.Y("Emotion"),
                        x=alt.X("Confidence", scale=alt.Scale(domain=[0, 1])),  # Fixed x-axis scale
                    ).properties(
                        width=300, # Chart width
                        height=200, # Chart height
                        padding={"left": 0, "top": 0, "right": 30, "bottom": 0},  # Compact padding
                    ).configure_axis(
                        labelFontSize=13,  # Smaller axis labels
                        titleFontSize=13,
                    )

                    st.altair_chart(chart, width='content')

                    # TODO make chart for faces that leaves frame disapear

    else:
        emotion_placeholder.error(body="No face detected", icon="❌")

    if show_nb_faces:
        with nb_faces_placeholder.container():
            st.container(border=True).metric("Number of detected faces", face_id+1)


    if show_fps:
        with fps_placeholder.container():
            st.container(border=True).metric("FPS", current_fps)
