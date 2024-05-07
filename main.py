import os
import shutil
import tempfile
from datetime import datetime

import cv2
import streamlit as st
from predict2 import predict_frames_from_folder

if not os.path.exists("frames"):
    os.makedirs("frames")

def save_frame(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join("frames", f"frame_{timestamp}.jpg")
    cv2.imwrite(filename, frame)

st.set_page_config(page_title="مراقب", layout="wide")
st.title("مراقب")

if "capture" not in st.session_state:
    st.session_state.capture = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

prediction_text = ""

message_placeholder = st.empty()
video_container = st.empty()

col1, col2, col3 = st.columns(3)

with st.sidebar:
    st.title("Realtime Violence Detection")

with col1:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        st.session_state.capture = True
        st.session_state.frame_count = 0
        st.session_state.cap = cv2.VideoCapture(video_path)
        st.video(video_path)
        prediction_placeholder = st.empty()



dir_path = "frames"

if st.session_state.capture:
    while st.session_state.capture:
        ret, frame = st.session_state.cap.read()
        if not ret:
            message_placeholder.error("Failed to capture video.")
            st.session_state.capture = False
            if hasattr(st.session_state, "cap"):
                st.session_state.cap.release()
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # video_container.image(frame_rgb, channels="RGB", use_column_width=True)

        save_frame(frame)
        st.session_state.frame_count += 1

        if st.session_state.frame_count % 100 == 0:
            frames_folder_path = "frames"
            prediction = predict_frames_from_folder(frames_folder_path, "model.h5")
            prediction_text = prediction
            prediction_placeholder.markdown("### Prediction Result")
            prediction_placeholder.write(prediction)
            st.session_state.last_prediction = prediction

if prediction_text:
    st.sidebar.markdown("### Last Prediction Result")
    st.sidebar.markdown(":blue["+prediction_text+"]")
else:
    st.sidebar.markdown("### Last Prediction Result")
    st.sidebar.markdown(":green[No Prediction]")
