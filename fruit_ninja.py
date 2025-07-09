import cv2
import mediapipe as mp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
import time
import random
from collections import deque

class HandGestureNetworkController:
    def __init__(self):
        # Initialize MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Network parameters
        self.G = nx.random_geometric_graph(20, 0.3)
        self.pos = nx.spring_layout(self.G, seed=42)
        self.selected_node = None
        self.highlighted_path = []
        self.node_colors = ['#1f78b4'] * len(self.G.nodes)
        self.node_sizes = [300] * len(self.G.nodes)
        
        # Parameters for hand control
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.rotation = 0
        self.last_pinch_distance = None
        self.last_hand_center = None
        self.is_pinching = False
        self.pinch_start_time = 0
        
        # For traffic animation
        self.traffic = {}  # (source, target) -> position (0-1)
        self.active_node = None
        self.last_active_time = time.time()
        
        # For pathfinding animation
        self.path_finding_active = False
        self.path_queue = deque()
        self.visited_nodes = set()
        self.shortest_paths = {}
        self.compute_all_shortest_paths()
        
        # For analytics
        self.node_activity = {node: 0 for node in self.G.nodes}
        self.edge_traffic = {edge: 0 for edge in self.G.edges}
        
        # Visualization settings
        self.background_color = (240, 240, 240)
        self.node_color = (31, 120, 180)
        self.edge_color = (160, 160, 160)
        self.highlight_color = (255, 69, 0)
        self.text_color = (0, 0, 0)
        
        # Network traffic thread
        self.traffic_thread = threading.Thread(target=self.update_network_traffic)
        self.traffic_thread.daemon = True
        self.running = True
        self.traffic_thread.start()
    
    def compute_all_shortest_paths(self):
        """Precompute shortest paths between all node pairs"""
        for source in self.G.nodes:
            self.shortest_paths[source] = {}
            for target in self.G.nodes:
                if source != target:
                    try:
                        path = nx.shortest_path(self.G, source=source, target=target)
                        self.shortest_paths[source][target] = path
                    except nx.NetworkXNoPath:
                        self.shortest_paths[source][target] = []
    
    def update_network_traffic(self):
        """Background thread to simulate network traffic"""
        while self.running:
            # Randomly generate new traffic
            if random.random() < 0.2 and len(self.traffic) < 10:
                source = random.choice(list(self.G.nodes))
                neighbors = list(self.G.neighbors(source))
                if neighbors:
                    target = random.choice(neighbors)
                    self.traffic[(source, target)] = 0.0
                    self.edge_traffic[(source, target)] = self.edge_traffic.get((source, target), 0) + 1
            
            # Update existing traffic
            traffic_to_remove = []
            for (source, target), pos in self.traffic.items():
                self.traffic[(source, target)] += 0.05
                if self.traffic[(source, target)] >= 1.0:
                    traffic_to_remove.append((source, target))
                    self.node_activity[target] += 1
            
            # Remove completed traffic
            for item in traffic_to_remove:
                del self.traffic[item]
            
            time.sleep(0.05)
    
    def detect_hand_landmarks(self, frame):
        """Detect hand landmarks using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    hand_points.append((x, y))
                landmarks.append(hand_points)
        
        return frame, landmarks
    
    def process_gestures(self, landmarks):
        """Process hand landmarks to detect gestures"""
        if not landmarks:
            self.is_pinching = False
            self.last_pinch_distance = None
            self.last_hand_center = None
            return
        
        # Process the first detected hand
        hand_points = landmarks[0]
        
        # Get thumb tip and index finger tip
        thumb_tip = hand_points[4]
        index_tip = hand_points[8]
        middle_tip = hand_points[12]
        
        # Calculate distance between thumb and index finger (pinch gesture)
        pinch_distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        # Calculate hand center (for panning)
        hand_center = np.mean(hand_points, axis=0).astype(int)
        
        # Detect pinch gesture (thumb and index finger close together)
        pinch_threshold = 30
        if pinch_distance < pinch_threshold:
            if not self.is_pinching:
                self.is_pinching = True
                self.pinch_start_time = time.time()
            else:
                # Long pinch activates pathfinding
                if time.time() - self.pinch_start_time > 1.0 and not self.path_finding_active:
                    self.trigger_pathfinding()
            
            # Zoom with pinch gesture (move hand forward/backward)
            if self.last_pinch_distance is not None:
                # Using y-coordinate for zoom (moving hand up/down)
                zoom_factor = (hand_center[1] - self.last_hand_center[1]) / 200.0
                self.zoom_level = max(0.5, min(2.0, self.zoom_level - zoom_factor))
        else:
            self.is_pinching = False
            
            # Pan the network with open hand
            if self.last_hand_center is not None:
                dx = (hand_center[0] - self.last_hand_center[0]) / 50.0
                dy = (hand_center[1] - self.last_hand_center[1]) / 50.0
                self.pan_offset[0] += dx
                self.pan_offset[1] += dy
        
        # Rotation using angle between thumb and middle finger
        if len(hand_points) > 12:
            wrist = hand_points[0]
            thumb_middle_vector = (thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
            middle_vector = (middle_tip[0] - wrist[0], middle_tip[1] - wrist[1])
            
            # Calculate angle between vectors
            dot_product = thumb_middle_vector[0] * middle_vector[0] + thumb_middle_vector[1] * middle_vector[1]
            mag1 = np.sqrt(thumb_middle_vector[0]**2 + thumb_middle_vector[1]**2)
            mag2 = np.sqrt(middle_vector[0]**2 + middle_vector[1]**2)
            
            if mag1 * mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                angle = np.arccos(cos_angle)
                
                # Check direction of rotation using cross product
                cross_product = thumb_middle_vector[0] * middle_vector[1] - thumb_middle_vector[1] * middle_vector[0]
                if cross_product < 0:
                    angle = -angle
                
                # Smooth rotation
                self.rotation = angle * 0.1
        
        # Update last values
        self.last_pinch_distance = pinch_distance
        self.last_hand_center = hand_center
    
    def trigger_pathfinding(self):
        """Start pathfinding animation from a random source to random targets"""
        self.path_finding_active = True
        self.path_queue.clear()
        self.visited_nodes = set()
        
        # Select a random source node
        source = random.choice(list(self.G.nodes))
        self.active_node = source
        
        # Queue up paths to 3 random destinations
        targets = random.sample(list(self.G.nodes), min(3, len(self.G.nodes)))
        for target in targets:
            if target != source and source in self.shortest_paths and target in self.shortest_paths[source]:
                path = self.shortest_paths[source][target]
                if path:
                    self.path_queue.append(path)
    
    def update_pathfinding(self):
        """Update pathfinding animation"""
        if not self.path_finding_active:
            return
            
        current_time = time.time()
        if current_time - self.last_active_time > 0.5:  # Move every 0.5 seconds
            self.last_active_time = current_time
            
            # If we have an active path
            if self.path_queue and self.active_node is not None:
                current_path = self.path_queue[0]
                current_index = current_path.index(self.active_node) if self.active_node in current_path else -1
                
                if current_index >= 0 and current_index < len(current_path) - 1:
                    # Move to next node in path
                    next_node = current_path[current_index + 1]
                    self.active_node = next_node
                    self.visited_nodes.add(next_node)
                    
                    # Add traffic animation
                    self.traffic[(current_path[current_index], next_node)] = 0.0
                else:
                    # Finished current path
                    self.path_queue.popleft()
                    
                    if not self.path_queue:
                        self.path_finding_active = False
                        self.active_node = None
            else:
                self.path_finding_active = False
                self.active_node = None
    
    def render_network(self, width=800, height=600):
        """Render the network graph with current state"""
        # Create figure with transparent background
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')
        
        # Apply transformations to node positions
        transformed_pos = {}
        zoom_matrix = np.array([[self.zoom_level, 0], [0, self.zoom_level]])
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(self.rotation), -np.sin(self.rotation)],
            [np.sin(self.rotation), np.cos(self.rotation)]
        ])
        
        for node in self.G.nodes:
            pos = np.array([self.pos[node][0], self.pos[node][1]])
            
            # Apply transformations: zoom, rotate, pan
            pos = zoom_matrix @ pos
            pos = rotation_matrix @ pos
            pos[0] += self.pan_offset[0] / 100.0
            pos[1] += self.pan_offset[1] / 100.0
            
            transformed_pos[node] = pos
        
        # Update node colors and sizes based on activity
        node_colors = []
        node_sizes = []
        for node in self.G.nodes:
            # Highlight active node and visited nodes
            if node == self.active_node:
                node_colors.append('red')
                node_sizes.append(500)
            elif node in self.visited_nodes:
                node_colors.append('orange')
                node_sizes.append(400)
            else:
                # Color based on activity level
                activity = min(1.0, self.node_activity.get(node, 0) / 10.0)
                r = int(31 + activity * 224)
                g = int(120 - activity * 51)
                b = int(180 - activity * 100)
                node_colors.append(f'#{r:02x}{g:02x}{b:02x}')
                node_sizes.append(300 + activity * 200)
        
        # Draw edges
        for u, v in self.G.edges():
            x1, y1 = transformed_pos[u]
            x2, y2 = transformed_pos[v]
            
            # Get edge traffic intensity
            traffic_intensity = min(1.0, self.edge_traffic.get((u, v), 0) / 20.0)
            edge_color = (0.6 - traffic_intensity * 0.4, 0.6 - traffic_intensity * 0.4, 0.6)
            edge_width = 1 + traffic_intensity * 3
            
            # Draw the edge
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width, alpha=0.7, zorder=1)
            
            if (u, v) in self.traffic:
                pos = self.traffic[(u, v)]
                packet_x = x1 + (x2 - x1) * pos
                packet_y = y1 + (y2 - y1) * pos
                ax.scatter(packet_x, packet_y, color='red', s=80, zorder=3)
        
        nx.draw_networkx_nodes(
            self.G, transformed_pos, 
            node_color=node_colors, 
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )
        
        label_pos = {node: (pos[0], pos[1] + 0.05) for node, pos in transformed_pos.items()}
        nx.draw_networkx_labels(
            self.G, label_pos, 
            font_size=10, 
            font_color='black',
            ax=ax
        )
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        mat = np.array(canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        return img
    
    def draw_overlay_info(self, frame, hand_detected):
        """Draw overlay information on the frame"""
        h, w, _ = frame.shape
        
        zoom_text = f"Zoom: {self.zoom_level:.2f}x"
        cv2.putText(frame, zoom_text, (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, zoom_text, (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        
       