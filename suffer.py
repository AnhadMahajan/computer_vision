import cv2
import numpy as np
import mediapipe as mp
import time

class SubwaySurfersHandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.left_region = None
        self.center_region = None
        self.right_region = None

        self.hand_state = "center"
        self.is_jumping = False
        self.is_rolling = False
        self.jump_start_time = 0
        self.roll_start_time = 0

        self.JUMP_DURATION = 0.7
        self.ROLL_DURATION = 0.7
        self.HAND_Y_THRESHOLD = 0.4

        self.frame_width = None
        self.frame_height = None

    def setup_regions(self, frame_width, frame_height):
        region_width = frame_width // 3
        self.left_region = (0, region_width)
        self.center_region = (region_width, 2 * region_width)
        self.right_region = (2 * region_width, frame_width)
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_hand_position(self, hand_landmarks):
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        x = (index_mcp.x + middle_mcp.x + wrist.x) / 3
        y = (index_mcp.y + middle_mcp.y + wrist.y) / 3
        return (x, y)

    def determine_lane_position(self, hand_x):
        pixel_x = int(hand_x * self.frame_width)
        if pixel_x < self.left_region[1]:
            return "left"
        elif pixel_x > self.right_region[0]:
            return "right"
        else:
            return "center"

    def determine_vertical_action(self, hand_y, prev_hand_y):
        if prev_hand_y is None:
            return None
        y_movement = hand_y - prev_hand_y
        threshold = 0.05
        if hand_y < self.HAND_Y_THRESHOLD and y_movement < -threshold:
            return "jump"
        elif hand_y > 1 - self.HAND_Y_THRESHOLD and y_movement > threshold:
            return "roll"
        return None

    def draw_control_visualization(self, frame):
        cv2.line(frame, (self.left_region[1], 0), (self.left_region[1], self.frame_height), (0, 255, 0), 2)
        cv2.line(frame, (self.right_region[0], 0), (self.right_region[0], self.frame_height), (0, 255, 0), 2)
        jump_line_y = int(self.HAND_Y_THRESHOLD * self.frame_height)
        roll_line_y = int((1 - self.HAND_Y_THRESHOLD) * self.frame_height)
        cv2.line(frame, (0, jump_line_y), (self.frame_width, jump_line_y), (255, 0, 0), 2)
        cv2.line(frame, (0, roll_line_y), (self.frame_width, roll_line_y), (255, 0, 0), 2)
        cv2.putText(frame, "LEFT", (self.left_region[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "CENTER", (self.center_region[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "RIGHT", (self.right_region[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "JUMP ZONE", (10, jump_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "ROLL ZONE", (10, roll_line_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        position_text = f"Lane: {self.hand_state.upper()}"
        if self.is_jumping:
            action_text = "Action: JUMPING"
            action_color = (0, 255, 255)
        elif self.is_rolling:
            action_text = "Action: ROLLING"
            action_color = (255, 0, 255)
        else:
            action_text = "Action: RUNNING"
            action_color = (255, 255, 255)
        cv2.putText(frame, position_text, (10, self.frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, action_text, (10, self.frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        cv2.putText(frame, "Move hand LEFT/RIGHT to change lanes", (10, self.frame_height - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Move hand quickly UP to JUMP", (10, self.frame_height - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Move hand quickly DOWN to ROLL", (10, self.frame_height - 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        if not success:
            print("Failed to open camera")
            return
        self.setup_regions(frame.shape[1], frame.shape[0])
        prev_hand_position = None
        prev_hand_y = None
        print("=" * 50)
        print("Subway Surfers Hand Control Started")
        print("=" * 50)
        print("Instructions:")
        print("1. Position yourself in front of the camera")
        print("2. Move your hand LEFT or RIGHT to change lanes")
        print("3. Move your hand quickly UP to JUMP")
        print("4. Move your hand quickly DOWN to ROLL")
        print("5. Press 'q' to quit")
        print("=" * 50)
        print("Starting in 3 seconds...")
        time.sleep(3)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            current_time = time.time()
            if self.is_jumping and current_time - self.jump_start_time > self.JUMP_DURATION:
                self.is_jumping = False
            if self.is_rolling and current_time - self.roll_start_time > self.ROLL_DURATION:
                self.is_rolling = False
            self.draw_control_visualization(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    hand_x, hand_y = self.get_hand_position(hand_landmarks)
                    px = int(hand_x * self.frame_width)
                    py = int(hand_y * self.frame_height)
                    if self.is_jumping:
                        color = (0, 255, 255)
                    elif self.is_rolling:
                        color = (255, 0, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.circle(frame, (px, py), 15, color, -1)
                    new_lane = self.determine_lane_position(hand_x)
                    if new_lane != self.hand_state:
                        print(f"Changed lane to: {new_lane}")
                        self.hand_state = new_lane
                    if not self.is_jumping and not self.is_rolling:
                        vertical_action = self.determine_vertical_action(hand_y, prev_hand_y)
                        if vertical_action == "jump":
                            self.is_jumping = True
                            self.jump_start_time = current_time
                            print("JUMP!")
                        elif vertical_action == "roll":
                            self.is_rolling = True
                            self.roll_start_time = current_time
                            print("ROLL!")
                    prev_hand_y = hand_y
                    prev_hand_position = (px, py)
            else:
                prev_hand_position = None
                prev_hand_y = None
            cv2.imshow('Subway Surfers Hand Control', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = SubwaySurfersHandController()
    controller.run()