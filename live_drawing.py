import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import io
import base64
from datetime import datetime
import json


st.set_page_config(
    page_title="AI Drawing Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .control-panel {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .drawing-area {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    
    .gesture-info {
        background: linear-gradient(135deg, #ff9a56, #ffad56);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .btn-custom {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1.5rem;
        border: none;
        border-radius: 25px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .btn-custom:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return hands, mp_hands, mp_drawing

class GestureRecognizer:
    def __init__(self):
        self.prev_landmarks = None
        
    def get_gesture(self, landmarks):
        if not landmarks:
            return "no_hand"
            
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        
        fingers_up = []
        
        if thumb_tip.x > landmarks[3].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), 
                         (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
            if tip.y < pip.y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        if fingers_up == [0, 1, 0, 0, 0]:
            return "draw"
        elif fingers_up == [0, 1, 1, 0, 0]:
            return "erase"
        elif sum(fingers_up) == 5:
            return "clear"
        elif fingers_up == [1, 0, 0, 0, 0]:
            return "select"
        else:
            return "idle"

class DrawingCanvas:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.drawing_points = []
        self.brush_size = 5
        self.brush_color = (0, 0, 255)  # BGR format
        
    def add_point(self, x, y, gesture):
        if gesture == "draw":
            self.drawing_points.append((int(x), int(y), "draw"))
        elif gesture == "erase":
            self.drawing_points.append((int(x), int(y), "erase"))
            
    def update_canvas(self):
        if len(self.drawing_points) > 1:
            for i in range(1, len(self.drawing_points)):
                if self.drawing_points[i][2] == "draw" and self.drawing_points[i-1][2] == "draw":
                    cv2.line(self.canvas, 
                           (self.drawing_points[i-1][0], self.drawing_points[i-1][1]),
                           (self.drawing_points[i][0], self.drawing_points[i][1]),
                           self.brush_color, self.brush_size)
                elif self.drawing_points[i][2] == "erase":
                    cv2.circle(self.canvas, 
                             (self.drawing_points[i][0], self.drawing_points[i][1]),
                             self.brush_size * 2, (255, 255, 255), -1)
    
    def clear_canvas(self):
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.drawing_points = []
        
    def get_canvas(self):
        return self.canvas.copy()

if 'canvas' not in st.session_state:
    st.session_state.canvas = DrawingCanvas()
if 'gesture_recognizer' not in st.session_state:
    st.session_state.gesture_recognizer = GestureRecognizer()
if 'drawing_stats' not in st.session_state:
    st.session_state.drawing_stats = {'strokes': 0, 'time_drawing': 0}

st.markdown("""
<div class="main-header">
    <h1>🎨 AI Drawing Studio</h1>
    <p style="color: white; font-size: 1.2rem; margin: 0;">Draw in the air with your hands!</p>
</div>
""", unsafe_allow_html=True)

hands, mp_hands, mp_drawing = initialize_mediapipe()

with st.sidebar:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Controls")
    
    camera_enabled = st.checkbox("📹 Enable Camera", value=True)
    
    st.markdown("### 🖌️ Brush Settings")
    brush_size = st.slider("Brush Size", 1, 20, 5)
    brush_color = st.color_picker("Brush Color", "#FF0000")
    
    hex_color = brush_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    
    st.session_state.canvas.brush_size = brush_size
    st.session_state.canvas.brush_color = bgr_color
    
    st.markdown("### 🎯 Actions")
    if st.button("🗑️ Clear Canvas", key="clear_btn"):
        st.session_state.canvas.clear_canvas()
        st.rerun()
    
    if st.button("💾 Save Drawing", key="save_btn"):
        canvas_img = st.session_state.canvas.get_canvas()
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
        
        buf = io.BytesIO()
        canvas_pil.save(buf, format='PNG')
        st.session_state.saved_image = buf.getvalue()
        st.success("Drawing saved!")
    
    if 'saved_image' in st.session_state:
        st.download_button(
            label="⬇️ Download Drawing",
            data=st.session_state.saved_image,
            file_name=f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
    
    st.markdown("### 🤏 Gesture Guide")
    gestures = {
        "👆 Index Finger": "Draw",
        "✌️ Peace Sign": "Erase", 
        "🖐️ Open Hand": "Clear Canvas",
        "👍 Thumbs Up": "Select Tool"
    }
    
    for gesture, action in gestures.items():
        st.markdown(f"**{gesture}**: {action}")
    
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="drawing-area">', unsafe_allow_html=True)
    
    if camera_enabled:
        camera_placeholder = st.empty()
        canvas_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check your camera permissions.")
        else:
            frame_count = 0
            current_gesture = "idle"
            
            with st.container():
                while camera_enabled and frame_count < 300:  
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    frame = cv2.flip(frame, 1)  # Mirror the frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    results = hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            current_gesture = st.session_state.gesture_recognizer.get_gesture(
                                hand_landmarks.landmark)
                            
                            if len(hand_landmarks.landmark) > 8:
                                index_tip = hand_landmarks.landmark[8]
                                x = int(index_tip.x * st.session_state.canvas.width)
                                y = int(index_tip.y * st.session_state.canvas.height)
                                
                                if current_gesture in ["draw", "erase"]:
                                    st.session_state.canvas.add_point(x, y, current_gesture)
                                elif current_gesture == "clear":
                                    st.session_state.canvas.clear_canvas()
                                
                                cv2.circle(frame, 
                                         (int(index_tip.x * frame.shape[1]), 
                                          int(index_tip.y * frame.shape[0])), 
                                         10, (0, 255, 0), -1)
                    
                    st.session_state.canvas.update_canvas()
                    
                    if frame_count % 3 == 0:
                        camera_placeholder.image(frame, channels="BGR", caption="📹 Camera Feed")
                        canvas_placeholder.image(
                            cv2.cvtColor(st.session_state.canvas.get_canvas(), cv2.COLOR_BGR2RGB),
                            caption="🎨 Your Drawing"
                        )
                    
                    if frame_count % 10 == 0:
                        break
            
            cap.release()
    else:
        st.info("📷 Enable camera to start drawing!")
        st.image(
            cv2.cvtColor(st.session_state.canvas.get_canvas(), cv2.COLOR_BGR2RGB),
            caption="🎨 Your Drawing Canvas"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Status Panel")
    
    st.markdown(f"""
    <div class="gesture-info">
        <strong>Current Gesture:</strong><br>
        {current_gesture.title().replace('_', ' ')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-card">
        <h4>Drawing Stats</h4>
        <p>Points: {len(st.session_state.canvas.drawing_points)}</p>
        <p>Brush Size: {brush_size}px</p>
    </div>
    """, unsafe_allow_html=True)
    
    status_color = "status-active" if camera_enabled else "status-inactive"
    status_text = "Active" if camera_enabled else "Inactive"
    
    st.markdown(f"""
    <div class="stats-card">
        <h4>Camera Status</h4>
        <p><span class="status-indicator {status_color}"></span>{status_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Pro Tips")
    tips = [
        "🖐️ Keep your hand visible to the camera",
        "💡 Use good lighting for better tracking",
        "🎯 Move slowly for precise drawing",
        "🔄 Try different gestures for various tools",
        "📏 Adjust brush size for different effects"
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎨 <strong>AI Drawing Studio</strong> - Draw with your hands using computer vision!</p>
    <p>Built with Streamlit, OpenCV, and MediaPipe</p>
</div>
""", unsafe_allow_html=True)