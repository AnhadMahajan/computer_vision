import cv2
import mediapipe as mp
import pyautogui
import time
from threading import Thread
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  
    min_detection_confidence=0.3,  
    min_tracking_confidence=0.3    
)
mp_drawing = mp.solutions.drawing_utils

JUMP_COOLDOWN = 0.2 
last_jump_time = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 60)  
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

action_queue = []
is_running = True

previous_hand_state = None  
hand_state_history = [] 
HISTORY_LENGTH = 3

last_frame_time = 0
fps = 0

def calculate_finger_curl(landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx):
    """Calculate how curled a finger is"""
    tip = landmarks.landmark[finger_tip_idx]
    pip = landmarks.landmark[finger_pip_idx]
    mcp = landmarks.landmark[finger_mcp_idx]
    
    vec_base = np.array([pip.x - mcp.x, pip.y - mcp.y])
    vec_finger = np.array([tip.x - pip.x, tip.y - pip.y])
    
    if np.linalg.norm(vec_base) == 0 or np.linalg.norm(vec_finger) == 0:
        return 0
    
    vec_base = vec_base / np.linalg.norm(vec_base)
    vec_finger = vec_finger / np.linalg.norm(vec_finger)
     
    curl = np.dot(vec_base, vec_finger)
    
    return curl

def determine_hand_state(landmarks):
    """More reliable hand state detection"""
    index_curl = calculate_finger_curl(
        landmarks, 
        mp_hands.HandLandmark.INDEX_FINGER_TIP, 
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP
    )
    
    middle_curl = calculate_finger_curl(
        landmarks, 
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP
    )
    
    ring_curl = calculate_finger_curl(
        landmarks, 
        mp_hands.HandLandmark.RING_FINGER_TIP, 
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_MCP
    )
    
    avg_curl = (index_curl + middle_curl + ring_curl) / 3
    
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    thumb_index_distance = ((thumb_tip.x - index_tip.x)**2 + 
                           (thumb_tip.y - index_tip.y)**2)**0.5
    
    if avg_curl > 0.5 and thumb_index_distance > 0.1:
        return "OPEN"
    elif avg_curl < -0.2:
        return "CLOSED"
    else:
        return "NEUTRAL"

def action_executor():
    """Separate thread to execute actions"""
    global action_queue, last_jump_time
    
    while is_running:
        if action_queue:
            action = action_queue.pop(0)
            current_time = time.time()
            
            if action == "JUMP" and current_time - last_jump_time > JUMP_COOLDOWN:
                pyautogui.press('space')
                last_jump_time = current_time
        
        time.sleep(0.01)

def get_smoothed_state(current_state):
    """Apply temporal smoothing to hand states"""
    global hand_state_history
    
    hand_state_history.append(current_state)
    
    if len(hand_state_history) > HISTORY_LENGTH:
        hand_state_history.pop(0)
    
    if len(hand_state_history) == HISTORY_LENGTH:
        if all(state == hand_state_history[0] for state in hand_state_history):
            return hand_state_history[0]
    
    if hand_state_history:
        states = {"OPEN": 0, "CLOSED": 0, "NEUTRAL": 0, None: 0}
        for state in hand_state_history:
            if state:
                states[state] += 1
        return max(states, key=states.get)
     
    return None

def process_frame(image):
    """Process a single frame to detect gestures"""
    global action_queue, previous_hand_state, fps, last_frame_time
    
    current_time = time.time()
    if last_frame_time:
        fps = 1 / (current_time - last_frame_time)
    last_frame_time = current_time
    
    small_image = cv2.resize(image, (160, 120))
    
    rgb_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
    
    rgb_image.flags.writeable = False
    results = hands.process(rgb_image)
    rgb_image.flags.writeable = True
    
    display_image = image.copy()
    
    status_text = "No hand detected"
    status_color = (100, 100, 100)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            for i, landmark in enumerate(hand_landmarks.landmark):
                hand_landmarks.landmark[i].x *= (display_image.shape[1] / small_image.shape[1])
                hand_landmarks.landmark[i].y *= (display_image.shape[0] / small_image.shape[0])
            
            mp_drawing.draw_landmarks(
                display_image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            )
            
            try:
                raw_hand_state = determine_hand_state(hand_landmarks)
                current_hand_state = get_smoothed_state(raw_hand_state)
                
                if current_hand_state == "OPEN":
                    status_text = "Hand OPEN"
                    status_color = (0, 255, 0)
                elif current_hand_state == "CLOSED":
                    status_text = "Hand CLOSED"
                    status_color = (0, 0, 255)
                else:
                    status_text = "Neutral"
                    status_color = (255, 165, 0)
                
                if previous_hand_state == "CLOSED" and current_hand_state == "OPEN":
                    action_queue.append("JUMP")
                    status_text = "JUMP!"
                    status_color = (255, 255, 0)
                
                previous_hand_state = current_hand_state
            except Exception as e:
                status_text = "Error: " + str(e)[:20]
                status_color = (0, 0, 255)
    else:
        previous_hand_state = None
    
    cv2.putText(display_image, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # cv2.putText(display_image, f"FPS: {fps:.1f}", (display_image.shape[1] - 100, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # cv2.putText(display_image, "Make a fist then open to JUMP", (10, display_image.shape[0] - 30), 
    #             cv2.F ONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display_image

def main():
    global frame_count, is_running
    
    action_thread = Thread(target=action_executor)
    action_thread.daemon = True
    action_thread.start()
    
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame from webcam")
                time.sleep(1)
                continue
            
            try:
                frame = cv2.flip(frame, 1)
                
                display_frame = process_frame(frame)
                    
                cv2.imshow('Fixed Dino Controller', display_frame)
            except Exception as e:  
                print(f"Error processing frame: {e}")
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        is_running = False
        cap.release()
        cv2.destroyAllWindows()
        print("Controller stopped")

if __name__ == "__main__":
    main()