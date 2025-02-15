import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

boxes = [
    {'x': 100, 'y': 100, 'size': 100, 'dragging': False},
    {'x': 300, 'y': 100, 'size': 100, 'dragging': False},
    {'x': 100, 'y': 300, 'size': 100, 'dragging': False},
    {'x': 300, 'y': 300, 'size': 100, 'dragging': False},
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            
            index_finger_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
            middle_finger_tip = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            
            distance = np.linalg.norm(np.array(index_finger_tip) - np.array(middle_finger_tip))
            
            for box in boxes:
                if distance < 30 and box['x'] < index_finger_tip[0] < box['x'] + box['size'] and box['y'] < index_finger_tip[1] < box['y'] + box['size']:
                    box['dragging'] = True
                elif distance >= 30:
                    box['dragging'] = False
            
            for box in boxes:
                if box['dragging']:
                    box['x'] = index_finger_tip[0] - box['size'] // 2
                    box['y'] = index_finger_tip[1] - box['size'] // 2
                    
    for box in boxes:
        cv2.rectangle(frame, (box['x'], box['y']), (box['x'] + box['size'], box['y'] + box['size']), (0, 255, 0), -1)
    
    cv2.imshow("Hand Tracking Boxes", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
