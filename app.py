"""
app.py

STEP 3 of the project — THIS is the file that becomes your live website.

WHAT THIS SCRIPT DOES (in plain English):
  1. Loads the trained model from train_classifier.py
  2. Defines one function: given a webcam frame, find the hand,
     extract landmarks, and predict which sign it is
  3. Gradio wraps that function in a web interface automatically —
     you get a working website with a webcam input and live
     predictions, without writing any HTML/CSS/JavaScript.

HOW DEPLOYMENT WORKS:
  When you upload this project to Hugging Face Spaces, their servers
  run this exact file. Hugging Face automatically installs
  requirements.txt, starts app.py, and gives you a public URL where
  ANYONE (including recruiters) can open it in their browser and use
  your live webcam sign-language recognizer. No separate "backend" or
  "frontend" needed — Gradio + this one file IS the whole website.

TO TEST LOCALLY FIRST:
  python app.py
  Then open the local URL it prints (usually http://127.0.0.1:7860)
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import gradio as gr

from utils import get_landmark_row

MODEL_PATH = "sign_classifier.joblib"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained model once, when the app starts (not on every frame —
# that would be slow)
model = joblib.load(MODEL_PATH)

# We create ONE Hands detector and reuse it for every frame, since
# creating a new one each time would be slow.
hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    static_image_mode=True,  # each Gradio frame is treated independently
)


def predict_sign(frame):
    """
    This is the core function Gradio calls every time it gets a new
    webcam frame.

    Input:  frame -> a webcam image (numpy array, RGB) from the browser
    Output: the same image, with the predicted sign drawn on it as text
    """
    if frame is None:
        return frame

    results = hands_detector.process(frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the hand skeleton on the image so users can see it working
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Extract the same 63 numbers the model was trained on
        row = get_landmark_row(hand_landmarks)
        row = np.array(row).reshape(1, -1)

        prediction = model.predict(row)[0]
        confidence = model.predict_proba(row).max()

        label_text = f"{prediction} ({confidence*100:.0f}%)"
        cv2.putText(
            frame, label_text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )
    else:
        cv2.putText(
            frame, "No hand detected", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )

    return frame


# --- Build the Gradio interface ---
# gr.Interface automatically builds a webpage: webcam input on the left,
# processed output on the right, refreshing continuously (live=True).
demo = gr.Interface(
    fn=predict_sign,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Your webcam"),
    outputs=gr.Image(label="Prediction"),
    live=True,
    title="Real-Time Sign Language to English Translator",
    description=(
        "Show a hand sign to your webcam. The app detects your hand "
        "landmarks with MediaPipe and classifies the sign using a "
        "Random Forest model trained on custom-collected data. "
        "Built with Python, OpenCV, MediaPipe, scikit-learn, and Gradio."
    ),
)

if __name__ == "__main__":
    demo.launch()
