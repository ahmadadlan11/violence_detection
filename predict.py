import os

import cv2
import numpy as np
import streamlit as st
from discord_webhook import sendMsg
from dotenv import load_dotenv
from tensorflow.keras.models import load_model  # type: ignore

load_dotenv()

WEBHOOK_SERVER_URL = os.getenv("WEBHOOK_SERVER_URL")


IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# model = load_model("model.h5")


def process_frames(
    frames_list, batch_number, MODEL_PATH="model.h5"
):
    """
    Process and predict the given list of frames, including the batch number in the output.
    """
    model = load_model(MODEL_PATH)
    # frames_array = np.array(frames_list)
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[
        0
    ]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]

    print(
        f"Batch {batch_number}: Predicted: {predicted_class_name}, Confidence: {predicted_labels_probabilities[predicted_label]:.4f}"
    )

    # print(WEBHOOK_SERVER_URL)

    if predicted_class_name == CLASSES_LIST[1]:
        sendMsg(WEBHOOK_SERVER_URL, CLASSES_LIST[1] + " Detected")
    st.write(
        f"Batch {batch_number}: Predicted: {predicted_class_name}, Confidence: {predicted_labels_probabilities[predicted_label]:.4f}"
    )


def predict_frames_from_folder(frames_folder_path, MODEL_PATH):
    frame_paths = [
        os.path.join(frames_folder_path, f)
        for f in os.listdir(frames_folder_path)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    frame_paths.sort()

    if not frame_paths:
        st.write("No frames available for prediction.")
        return

    frames_list = []
    batch_number = 1

    with st.sidebar:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_list.append(normalized_frame)

            if len(frames_list) == SEQUENCE_LENGTH:
                process_frames(frames_list, batch_number)
                frames_list = []
                batch_number += 1

    if len(frames_list) > 0:

        process_frames(frames_list, batch_number)