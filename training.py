import cv2
import numpy as np
import mediapipe as mp
import os
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
data = {label: [] for label in labels}

cap = cv2.VideoCapture(0)

for label in labels:
    print(f"Collecting data for: {label}. Press 'c' to capture, 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                data[label].append(landmarks)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, f"Collecting: {label}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Data Collection", frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'): 
            print(f"Captured for {label}: {len(data[label])} samples")
        elif key == ord('q'): 
            break

cap.release()
cv2.destroyAllWindows()

with open("sign_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("Data collection complete! Saved as 'sign_data.pkl'")
