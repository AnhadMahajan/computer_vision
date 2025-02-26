import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

blink_counter = 0
frame_counter = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    ear = None
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape
            
            try:
                left_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in LEFT_EYE]
                right_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in RIGHT_EYE]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < EYE_AR_THRESH:
                    frame_counter += 1
                else:
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_counter += 1
                        os.system('play -nq -t alsa synth 0.2 sine 1000') 
                    frame_counter = 0
                
                for point in left_eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
                for point in right_eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
            except:
                continue
    
    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    
    
    cv2.imshow("Eye Blink Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
