import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define keyboard layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

key_width = 60
key_height = 60
keyboard_origin = (50, 300)

# Draw the virtual keyboard
def draw_keyboard(frame):
    x, y = keyboard_origin
    for row in keys:
        for key in row:
            top_left = (x, y)
            bottom_right = (x + key_width, y + key_height)
            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
            cv2.putText(frame, key, (x + 15, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            x += key_width + 10
        x = keyboard_origin[0]
        y += key_height + 10

# Check if a finger is over a key
def check_key_press(landmark, frame):
    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
    x_offset, y_offset = keyboard_origin
    for row_index, row in enumerate(keys):
        for col_index, key in enumerate(row):
            key_x = x_offset + col_index * (key_width + 10)
            key_y = y_offset + row_index * (key_height + 10)
            if key_x <= x <= key_x + key_width and key_y <= y <= key_y + key_height:
                cv2.rectangle(frame, (key_x, key_y), (key_x + key_width, key_y + key_height), (0, 255, 0), -1)
                cv2.putText(frame, key, (key_x + 15, key_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                print(f"Key Pressed: {key}")
                return key
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    draw_keyboard(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            check_key_press(index_finger_tip, frame)

    cv2.imshow("Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
