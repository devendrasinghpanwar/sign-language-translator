"""
collect_data.py

STEP 1 of the project.

WHAT THIS SCRIPT DOES (in plain English):
  1. Opens your webcam
  2. Uses MediaPipe to find your hand and mark 21 points on it
     (fingertips, knuckles, wrist, etc.)
  3. Every time you press a LETTER key (a, b, c...), it saves the
     current hand position (as 21 x,y,z numbers) into a CSV file,
     labeled with that letter.
  4. You repeat this ~30-50 times per letter, showing slightly
     different hand angles each time, so the model learns to
     recognize the letter reliably (not just memorize one exact photo).

WHY LANDMARKS INSTEAD OF RAW IMAGES?
  A raw photo is huge (thousands of pixel values) and hard for a
  simple model to learn from. MediaPipe compresses your hand down to
  just 21 points x 3 coordinates = 63 numbers. That's small enough
  that even a basic classifier (no deep learning needed!) can learn
  the patterns.

CONTROLS:
  - Show a hand sign, then press the matching letter key (a-z) to save it
  - Press 'q' to quit and finish
"""

import cv2
import mediapipe as mp
import csv
import os

from utils import get_landmark_row

CSV_PATH = "hand_data.csv"

# --- Which letters/signs are we teaching the model? ---
# Start small! You can add more later once the pipeline works.
SIGNS_TO_COLLECT = ["a", "b", "c", "hello", "yes", "no"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    # Create the CSV file with a header row if it doesn't exist yet
    file_exists = os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, mode="a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = ["label"] + [f"coord_{i}" for i in range(63)]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    counts = {sign: 0 for sign in SIGNS_TO_COLLECT}

    print("Instructions:")
    print(f"  Show a sign, then press its first letter key to save it.")
    print(f"  Signs to collect: {SIGNS_TO_COLLECT}")
    print("  (For multi-letter signs like 'hello', press 'h', etc. "
          "— see key_map below)")
    print("  Press 'q' to quit.\n")

    # Map keyboard keys to sign labels (customize if signs share a first letter)
    key_map = {sign[0]: sign for sign in SIGNS_TO_COLLECT}

    with mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror image, feels more natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Show current counts on screen
            y_offset = 30
            for sign, count in counts.items():
                cv2.putText(
                    frame, f"{sign}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                y_offset += 25

            cv2.imshow("Data Collection - press a sign's key to save", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            pressed_char = chr(key) if 0 <= key < 256 else ""
            if pressed_char in key_map and results.multi_hand_landmarks:
                sign = key_map[pressed_char]
                row = get_landmark_row(results.multi_hand_landmarks[0])
                writer.writerow([sign] + row)
                counts[sign] += 1
                print(f"Saved sample for '{sign}' (total: {counts[sign]})")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nDone! Data saved to {CSV_PATH}")
    print("Next step: run train_classifier.py")


if __name__ == "__main__":
    main()
