"""
Optimized main application with better performance + 3D model viewer
"""
import cv2
import time
import numpy as np
from collections import deque
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import CONFIG
from models.detector import OptimizedObjectDetector, EnhancedMediaPipeDetector
from models.gesture_recognition import OptimizedGestureRecognizer
from models.activity_detection import ActivityDetector
from ar.overlay import draw_hand_landmark_labels
from ar.interaction import ARInteractionSystem
from utils.drawing import draw_3d_box, draw_label
from utils.display import ARDisplayManager

# -------------------------------------------------------------------------
# 3D MODEL MAP
# -------------------------------------------------------------------------
MODEL_MAP = {
    "person": "models/person.obj",
    "cell phone": "models/cell_phone.obj",
}

CLICK_X = None
CLICK_Y = None


def mouse_callback(event, x, y, flags, param):
    """Capture mouse click globally."""
    global CLICK_X, CLICK_Y
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_X = x
        CLICK_Y = y


# -------------------------------------------------------------------------
# 3D MODEL VIEWER
# -------------------------------------------------------------------------
def open_3d_model_async(path):
    threading.Thread(target=open_3d_model, args=(path,), daemon=True).start()


def open_3d_model(model_path):
    if not os.path.exists(model_path):
        print(f"[3D ERROR] Missing model: {model_path}")
        return

    try:
        mesh = o3d.io.read_triangle_mesh(model_path)
        if mesh.is_empty():
            print(f"[3D ERROR] Empty mesh: {model_path}")
            return

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        poly3d = [vertices[t] for t in triangles]
        collection = Poly3DCollection(poly3d, alpha=0.8, edgecolor="k")
        ax.add_collection3d(collection)

        ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
        ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
        ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(os.path.basename(model_path))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[3D VIEW ERROR] {e}")


class OptimizedARVisionSystem:
    """Optimized AR Vision System with better performance"""

    def __init__(self):
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.fps_history = deque(maxlen=60)

        # Performance optimization
        self.frame_skip = CONFIG['detection_settings']['frame_skip']
        self.last_processed_frame = 0
        self.performance_mode = 'balanced'  # 'fast', 'balanced', 'accurate'
        self.resolution_scale = 0.8  # Start with balanced scaling for better FPS
        self.target_fps = 15  # Target minimum FPS
        self.adaptive_mode = True  # Enable adaptive performance adjustment

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.detection_future = None

        # Initialize components
        self.object_detector = OptimizedObjectDetector()
        self.mediapipe_detector = EnhancedMediaPipeDetector()
        self.gesture_recognizer = OptimizedGestureRecognizer()
        self.activity_detector = ActivityDetector()
        self.ar_system = ARInteractionSystem()
        self.display_manager = None

        # Frame cache
        self.current_frame = None
        self.processed_frame = None
        self.detections = []
        self.mediapipe_results = {}
        self.gestures = []
        self.activities = []

        # MediaPipe caching for performance
        self.mediapipe_cache = {}
        self.last_mediapipe_frame_hash = None

        # Performance monitoring
        self.performance_stats = {
            'detection_time': deque(maxlen=20),
            'processing_time': deque(maxlen=20),
            'total_time': deque(maxlen=20),
            'frame_skipped': 0
        }

        # Add missing attributes
        self.total_frames = 0
        self.cache_hits = 0
        self.last_frame_time = None

    # ------------------------------------------------------------------
    # PERFORMANCE MODE
    # ------------------------------------------------------------------
    def apply_performance_mode(self):
        """Apply balanced performance mode settings for FPS + Accuracy"""
        if self.performance_mode == 'fast':
            self.frame_skip = 1  # Every frame for consistency
            self.resolution_scale = 0.67  # Moderate downscaling
            CONFIG['confidence_threshold'] = 0.4  # Higher for speed
        elif self.performance_mode == 'balanced':
            self.frame_skip = 1  # Every frame
            self.resolution_scale = 0.8  # Light downscaling
            CONFIG['confidence_threshold'] = 0.3  # Balanced threshold
        elif self.performance_mode == 'accurate':
            self.frame_skip = 1  # Every frame
            self.resolution_scale = 0.9  # Minimal downscaling
            CONFIG['confidence_threshold'] = 0.25  # High accuracy

    def print_welcome(self):
        """Print welcome message with performance info"""
        print("=" * 70)
        print("ENHANCED AR COMPUTER VISION SYSTEM")
        print(f"Device: {'GPU' if CONFIG['detection_settings']['use_gpu'] else 'CPU'}")
        print(f"Target Objects: {len(CONFIG['object_colors'])}")
        print(f"Performance Mode: {self.performance_mode}")
        print("=" * 70)
        print("\nFEATURES:")
        print("1. Balanced High-Accuracy Object Detection with YOLOv8m")
        print("2. Enhanced Animal Detection (Cat, Dog, Horse, Monkey, etc.)")
        print("3. Precise Fruit Detection (Apple, Banana, Orange, etc.)")
        print("4. Advanced Object Tracking for Continuous Detection")
        print("5. Optimized Parallel Processing with FPS + Accuracy Balance")
        print("\nCONTROLS:")
        print("  Q - Quit application")
        print("  R - Reset all interactions")
        print("  +/- - Adjust confidence threshold")
        print("  F - Toggle fullscreen")
        print("  SPACE - Take screenshot")
        print("  T - Toggle object tracking")
        print("  1 - Cycle performance modes (Fast/Balanced/Accurate)")
        print("  2 - Cycle resolution scales (1.0x/0.8x/0.67x)")
        print("  3 - Maximum accuracy mode (YOLOv8x + full resolution)")
        print("\nGESTURES:")
        print("  👌 PINCH - Select & rotate objects")
        print("  ✋ OPEN PALM - Scale objects")
        print("  ✌ VICTORY - Move objects")
        print("  👊 FIST - Reset transformations")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------------------
    def initialize_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(0)

        # Set optimal camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Try to enable hardware acceleration
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Test camera
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to initialize camera")

        # Create window and attach mouse callback for 3D model clicking
        cv2.namedWindow("Enhanced AR Vision System", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Enhanced AR Vision System", mouse_callback)

        print("✓ Camera initialized successfully")
        return True

    def capture_frame(self):
        """Capture frame with optimization"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # Resize if needed for performance
        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            new_width = 1280
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        return frame

    # ------------------------------------------------------------------
    # FRAME PROCESSING
    # ------------------------------------------------------------------
    def process_frame_parallel(self, frame):
        """Process frame in parallel threads with optimizations"""
        total_start = time.time()
        self.total_frames += 1

        # Convert to RGB for MediaPipe (in separate thread)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe caching for performance
        frame_hash = hash(rgb_frame.tobytes()) if self.resolution_scale >= 0.8 else None

        if frame_hash and frame_hash in self.mediapipe_cache:
            # Use cached MediaPipe results
            self.mediapipe_results = self.mediapipe_cache[frame_hash]
            self.cache_hits += 1
            mediapipe_time = 0.001  # Minimal time for cache hit
        else:
            # Process MediaPipe normally
            mediapipe_start = time.time()
            self.mediapipe_results = self.mediapipe_detector.process_frame(rgb_frame)
            mediapipe_time = time.time() - mediapipe_start

            # Cache results if high resolution
            if frame_hash and len(self.mediapipe_cache) < 10:  # Limit cache size
                self.mediapipe_cache[frame_hash] = self.mediapipe_results

        # Run object detection in parallel
        if self.detection_future is None or self.detection_future.done():
            # Submit new detection task
            self.detection_future = self.executor.submit(
                self.object_detector.detect_with_preprocess,
                frame,
                self.frame_count
            )

        # Get detection results if ready
        detection_start = time.time()
        if self.detection_future and self.detection_future.done():
            self.detections = self.detection_future.result()
        else:
            # Use cached detections if not ready
            self.detections = getattr(self.object_detector, "last_detections", [])
        detection_time = time.time() - detection_start

        # Process gestures and activities
        self.process_gestures_and_activities()

        # Calculate total processing time
        total_time = time.time() - total_start

        # Update performance stats
        self.performance_stats['detection_time'].append(detection_time)
        self.performance_stats['processing_time'].append(mediapipe_time)
        self.performance_stats['total_time'].append(total_time)

        return frame

    # ------------------------------------------------------------------
    # GESTURES / ACTIVITIES
    # ------------------------------------------------------------------
    def process_gestures_and_activities(self):
        """Process gestures and activities from MediaPipe results"""
        self.gestures = []
        self.activities = []

        # Process hand gestures
        if (self.mediapipe_results.get('hands') and
            hasattr(self.mediapipe_results['hands'], 'multi_hand_landmarks') and
                self.mediapipe_results['hands'].multi_hand_landmarks):

            for idx, hand_landmarks in enumerate(self.mediapipe_results['hands'].multi_hand_landmarks):
                handedness = None
                if (hasattr(self.mediapipe_results['hands'], 'multi_handedness') and
                        idx < len(self.mediapipe_results['hands'].multi_handedness)):
                    handedness = self.mediapipe_results['hands'].multi_handedness[idx]

                gesture = self.gesture_recognizer.recognize(hand_landmarks, handedness)
                if gesture:
                    self.gestures.append(gesture)

        # Process face activities
        if (self.mediapipe_results.get('face') and
            hasattr(self.mediapipe_results['face'], 'multi_face_landmarks') and
                self.mediapipe_results['face'].multi_face_landmarks):

            for face_landmarks in self.mediapipe_results['face'].multi_face_landmarks:
                face_activity = self.activity_detector.detect_face_activity(face_landmarks)
                if face_activity != "NEUTRAL":
                    self.activities.append(f"Face: {face_activity}")

        # Process pose activities
        if (self.mediapipe_results.get('pose') and
            hasattr(self.mediapipe_results['pose'], 'pose_landmarks') and
                self.mediapipe_results['pose'].pose_landmarks):

            pose_activity = self.activity_detector.detect_pose_activity(
                self.mediapipe_results['pose'].pose_landmarks
            )
            if pose_activity != "STANDING":
                self.activities.append(f"Pose: {pose_activity}")

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------
    def draw_frame(self, frame):
        """Draw all AR elements on frame"""
        height, width = frame.shape[:2]

        # Initialize display manager if needed
        if self.display_manager is None:
            self.display_manager = ARDisplayManager(width, height, CONFIG)

        # Draw detected objects
        frame = self.draw_objects(frame, width, height)

        # Draw hand landmarks
        frame = self.draw_hand_landmarks(frame)

        # Draw AR elements
        frame = self.draw_ar_elements(frame, width, height)

        # Update FPS
        self.update_fps()

        # Update UI
        frame = self.update_ui(frame, width, height)

        return frame

    def draw_objects(self, frame, width, height):
        """Draw detected objects with optimized rendering"""
        # Sort by confidence and limit for performance
        sorted_detections = sorted(self.detections, key=lambda x: x.get('confidence', 0), reverse=True)
        display_limit = min(CONFIG.get('max_panel_objects', 10), len(sorted_detections))

        for i, det in enumerate(sorted_detections[:display_limit]):
            # Check if det has required keys
            if 'bbox' not in det or 'name' not in det:
                continue

            # Skip very small objects for performance
            x1, y1, x2, y2 = det['bbox']
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area < 1000:  # Skip very small objects
                continue

            # Apply AR transformations if object is selected
            if (self.ar_system.selected_object and
                    self.ar_system.selected_object.get('name') == det['name']):
                transformed_bbox = self.ar_system.get_transformed_bbox(det['bbox'], width, height)
                x1, y1, x2, y2 = transformed_bbox
            else:
                x1, y1, x2, y2 = det['bbox']

            # Get object color
            color = CONFIG['object_colors'].get(det['name'], (0, 200, 255))

            # Draw 3D bounding box (thinner lines for performance)
            frame = draw_3d_box(frame, x1, y1, x2, y2, color, 1)

            # Draw label with confidence (only for top objects)
            if i < 8:  # Only label top 8 objects
                label = f"{det['name'].title()} ({det.get('confidence', 0):.2f})"
                if (self.ar_system.selected_object and
                        self.ar_system.selected_object.get('name') == det['name']):
                    label = f"⭐ {label}"

                # Position label
                label_y = max(30, y1 - 10)
                label_x = max(10, min(x1, width - 200))

                frame = draw_label(frame, label, (label_x, label_y),
                                   color, (255, 255, 255))

        return frame

    def draw_hand_landmarks(self, frame):
        """Draw hand landmarks with labels"""
        if (self.mediapipe_results.get('hands') and
            hasattr(self.mediapipe_results['hands'], 'multi_hand_landmarks') and
                self.mediapipe_results['hands'].multi_hand_landmarks):

            for idx, hand_landmarks in enumerate(self.mediapipe_results['hands'].multi_hand_landmarks):
                handedness = None
                if (hasattr(self.mediapipe_results['hands'], 'multi_handedness') and
                        idx < len(self.mediapipe_results['hands'].multi_handedness)):
                    handedness = self.mediapipe_results['hands'].multi_handedness[idx]

                hand_label = handedness.classification[0].label if handedness else "Hand"
                frame = draw_hand_landmark_labels(frame, hand_landmarks, hand_label)

        return frame

    def draw_ar_elements(self, frame, width, height):
        """Draw AR elements"""
        # Draw AR information panel
        if self.ar_system.selected_object and hasattr(self.ar_system, 'object_info_panel'):
            frame = self.ar_system.draw_info_panel(frame)

        # Draw activity information
        frame = self.draw_activity_info(frame)

        # Draw 3D model indicators for selected object
        if self.ar_system.selected_object:
            frame = self.draw_selected_object_indicators(frame, width, height)

        # Process gestures for AR interaction
        if self.gestures and self.detections:
            self.process_gestures_for_ar()

        return frame

    def draw_activity_info(self, frame):
        """Draw activity information for detected people"""
        for det in self.detections:
            if det.get('name') == 'person' and self.activities:
                # Find corresponding activity
                person_activity = None
                for activity in self.activities:
                    if 'Face:' in activity or 'Pose:' in activity:
                        person_activity = activity.split(': ')[1]
                        break

                if person_activity and person_activity != "NEUTRAL":
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    activity_x = max(10, x1)
                    activity_y = max(30, y1 - 40)

                    activity_text = f"Activity: {person_activity}"
                    frame = draw_label(frame, activity_text, (activity_x, activity_y),
                                       (50, 50, 150), (255, 255, 255), font_scale=0.6)
        return frame

    def draw_selected_object_indicators(self, frame, width, height):
        """Draw 3D model indicators for selected object"""
        for det in self.detections:
            if (self.ar_system.selected_object and
                    det.get('name') == self.ar_system.selected_object.get('name')):
                bbox = det.get('bbox', [0, 0, 0, 0])
                transformed_bbox = self.ar_system.get_transformed_bbox(bbox, width, height)
                frame = self.ar_system.draw_3d_model_indicators(frame, transformed_bbox)
                break
        return frame

    # ------------------------------------------------------------------
    # AR GESTURE CONTROL
    # ------------------------------------------------------------------
    def process_gestures_for_ar(self):
        """Process gestures for AR interaction"""
        for gesture in self.gestures:
            if gesture.get('confidence', 0) > 0.7:
                if gesture.get('name') == 'PINCH' and not self.ar_system.selected_object:
                    # Select the most confident object
                    if self.detections:
                        best_det = max(self.detections, key=lambda x: x.get('confidence', 0))
                        self.ar_system.select_object(best_det)
                else:
                    self.ar_system.process_gesture(gesture)

    # ------------------------------------------------------------------
    # FPS & UI
    # ------------------------------------------------------------------
    def update_fps(self):
        """Update FPS calculation and auto-adjust performance"""
        if self.last_frame_time:
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
            self.last_frame_time = current_time

            # Auto-adjust performance based on FPS
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            # Auto-adjust performance if FPS is too low
            if avg_fps < 15 and self.performance_mode != 'fast':
                print(f"Low FPS detected ({avg_fps:.1f}), switching to fast mode")
                self.performance_mode = 'fast'
                self.apply_performance_mode()
            elif avg_fps > 25 and self.performance_mode == 'fast':
                print(f"Good FPS detected ({avg_fps:.1f}), switching to balanced mode")
                self.performance_mode = 'balanced'
                self.apply_performance_mode()
        else:
            self.last_frame_time = time.time()

    def update_ui(self, frame, width, height):
        """Update all UI elements"""
        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        # Update display data
        detection_stats = self.object_detector.get_detection_stats() if hasattr(
            self.object_detector, 'get_detection_stats') else {
            "avg_time": 0.0,
            "active_tracks": 0
        }

        gesture_stats = self.gesture_recognizer.get_stats() if hasattr(
            self.gesture_recognizer, 'get_stats') else {
            "cache_hit_rate": "0%"
        }

        self.display_manager.update_display_data('system_status', [
            {'text': 'FPS', 'value': f'{avg_fps:.1f}', 'color': (255, 255, 0)},
            {'text': 'Detection Time', 'value': f'{detection_stats.get("avg_time", 0)*1000:.1f}ms',
             'color': (200, 200, 255)},
            {'text': 'Confidence', 'value': f'{CONFIG.get("confidence_threshold", 0.3):.2f}',
             'color': (200, 255, 200)},
            {'text': 'Active Tracks', 'value': str(detection_stats.get("active_tracks", 0)),
             'color': (255, 200, 200)},
            {'text': 'Gesture Cache', 'value': f'{gesture_stats.get("cache_hit_rate", "0%")}',
             'color': (255, 200, 255)}
        ])

        # Update detected objects
        object_display = []
        for i, det in enumerate(sorted(self.detections,
                                       key=lambda x: x.get('confidence', 0),
                                       reverse=True)[:8]):  # Show top 8

            # Add emoji for animals
            emoji = ""
            if det.get('name') in ['cat', 'dog', 'horse', 'bird', 'monkey']:
                emoji = "🐱" if det['name'] == 'cat' else \
                        "🐶" if det['name'] == 'dog' else \
                        "🐴" if det['name'] == 'horse' else \
                        "🐦" if det['name'] == 'bird' else \
                        "🐵" if det['name'] == 'monkey' else ""

            object_display.append({
                'text': f'{emoji}{det.get("name", "Unknown").title()}',
                'value': f'{det.get("confidence", 0):.2f}',
                'color': (0, 255, 0) if i == 0 else (180, 180, 255)
            })

        if not object_display:
            object_display.append({
                'text': 'No objects',
                'value': 'detected',
                'color': (180, 180, 180)
            })

        self.display_manager.update_display_data('detected_objects', object_display)

        # Update gestures
        gesture_display = []
        for gesture in self.gestures[:3]:  # Show top 3 gestures
            gesture_display.append({
                'text': f'{gesture.get("hand", "Unknown")} Hand',
                'value': f'{gesture.get("name", "Unknown")} ({gesture.get("confidence", 0):.2f})',
                'color': (255, 100, 255)
            })

        self.display_manager.update_display_data('gestures', gesture_display)

        # Update activities
        activity_display = []
        for i, activity in enumerate(self.activities[:3]):
            activity_display.append({
                'text': f'Activity {i+1}',
                'value': activity,
                'color': (200, 0, 255)
            })

        self.display_manager.update_display_data('activities', activity_display)

        # Update AR status
        ar_status = []
        if self.ar_system.selected_object:
            ar_status.append({
                'text': 'Selected Object',
                'value': self.ar_system.selected_object.get('name', 'Unknown').title(),
                'color': (0, 255, 255)
            })

            if self.ar_system.interaction_mode:
                ar_status.append({
                    'text': 'Interaction',
                    'value': self.ar_system.interaction_mode,
                    'color': (255, 255, 0)
                })

        self.display_manager.update_display_data('ar_status', ar_status)

        # Draw UI elements
        frame = self.display_manager.draw_control_panel(frame)

        # Draw HUD
        activity_summary = self.activities[0].split(": ")[1] if self.activities else "No activity"
        frame = self.display_manager.draw_hud(frame, avg_fps, len(self.detections), activity_summary)

        # Draw performance overlay
        frame = self.draw_performance_overlay(frame, width)

        return frame

    def draw_performance_overlay(self, frame, width):
        """Draw performance overlay"""
        if len(self.performance_stats['total_time']) > 0:
            avg_total = np.mean(self.performance_stats['total_time']) * 1000
            avg_detect = np.mean(self.performance_stats['detection_time']) * 1000
            avg_process = np.mean(self.performance_stats['processing_time']) * 1000

            perf_text = f"Total: {avg_total:.1f}ms | Detect: {avg_detect:.1f}ms | Process: {avg_process:.1f}ms"

            # Draw at bottom right
            text_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            x = width - text_size[0] - 10
            y = frame.shape[0] - 70

            frame = draw_label(frame, perf_text, (x, y),
                               (40, 40, 40), (200, 200, 200),
                               font_scale=0.4, thickness=1)

        return frame

    def draw_instruction_bar(self, frame, width, height):
        """Draw enhanced instruction bar"""
        cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, height - 50), (width, height), (50, 50, 50), 1)

        instructions = [
            "Q:Quit",
            "R:Reset",
            "+/-:Confidence",
            "F:Fullscreen",
            "SPACE:Screenshot",
            "T:Tracking",
            "PINCH:Select",
            "OPEN_PALM:Scale",
            "Click:3D Model"
        ]

        x_pos = 10
        for instruction in instructions:
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, instruction, (x_pos, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            x_pos += text_size[0] + 15

        return frame

    # ------------------------------------------------------------------
    # 3D MODEL CLICK HANDLING
    # ------------------------------------------------------------------
    def check_click_model(self):
        """Check if user clicked on any detected object and open 3D model."""
        global CLICK_X, CLICK_Y
        if CLICK_X is None:
            return

        x, y = CLICK_X, CLICK_Y
        CLICK_X = CLICK_Y = None

        for det in self.detections:
            if "bbox" not in det or "name" not in det:
                continue

            x1, y1, x2, y2 = det["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                name = det["name"].lower()
                if name in MODEL_MAP:
                    print(f"[3D] Loading model: {MODEL_MAP[name]}")
                    open_3d_model_async(MODEL_MAP[name])
                else:
                    print("[3D] No model for:", name)
                break

    # ------------------------------------------------------------------
    # INPUT HANDLING
    # ------------------------------------------------------------------
    def handle_keyboard_input(self, key):
        """Enhanced keyboard input handling"""
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.ar_system.reset()
            print("All interactions reset")
        elif key == ord('+') or key == ord('='):
            CONFIG['confidence_threshold'] = min(0.95, CONFIG.get('confidence_threshold', 0.3) + 0.05)
            print(f"Confidence threshold: {CONFIG['confidence_threshold']:.2f}")
        elif key == ord('-') or key == ord('_'):
            CONFIG['confidence_threshold'] = max(0.1, CONFIG.get('confidence_threshold', 0.3) - 0.05)
            print(f"Confidence threshold: {CONFIG['confidence_threshold']:.2f}")
        elif key == ord('f'):
            is_fullscreen = cv2.getWindowProperty("Enhanced AR Vision System",
                                                  cv2.WND_PROP_FULLSCREEN)
            if is_fullscreen == 0:
                cv2.setWindowProperty("Enhanced AR Vision System",
                                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen enabled")
            else:
                cv2.setWindowProperty("Enhanced AR Vision System",
                                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Window mode")
        elif key == ord('t'):
            tracking_enabled = CONFIG['detection_settings'].get('tracking_enabled', False)
            CONFIG['detection_settings']['tracking_enabled'] = not tracking_enabled
            status = "enabled" if CONFIG['detection_settings']['tracking_enabled'] else "disabled"
            print(f"Object tracking {status}")
        elif key == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ar_screenshot_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('1'):
            # Cycle performance modes
            modes = ['fast', 'balanced', 'accurate']
            current_idx = modes.index(self.performance_mode)
            self.performance_mode = modes[(current_idx + 1) % len(modes)]
            self.apply_performance_mode()
            print(f"Performance mode: {self.performance_mode} "
                  f"(Frame skip: {self.frame_skip}, Resolution: {self.resolution_scale}x)")
        elif key == ord('2'):
            # Cycle resolution scales
            scales = [1.0, 0.75, 0.5]
            if self.resolution_scale not in scales:
                self.resolution_scale = 0.8  # fallback
            current_idx = scales.index(self.resolution_scale)
            self.resolution_scale = scales[(current_idx + 1) % len(scales)]
            print(f"Resolution scale: {self.resolution_scale}x")

        return True

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        """Main optimized application loop"""
        self.print_welcome()

        try:
            self.initialize_camera()
        except Exception as e:
            print(f"✗ Failed to initialize camera: {e}")
            return

        self.running = True
        print("System ready. Starting Enhanced AR Vision System...\n")
        print("Tip: Click on detected objects that have 3D models to open them.")

        while self.running:
            frame_start = time.time()

            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                break

            # Process frame (parallel)
            self.current_frame = self.process_frame_parallel(frame)

            # Draw frame
            self.current_frame = self.draw_frame(self.current_frame)

            # Handle 3D model click
            self.check_click_model()

            # Draw instruction bar
            height, width = self.current_frame.shape[:2]
            self.current_frame = self.draw_instruction_bar(self.current_frame, width, height)

            # Display frame
            cv2.imshow("Enhanced AR Vision System", self.current_frame)

            # Update frame count
            self.frame_count += 1

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.running = self.handle_keyboard_input(key)

            # Print stats occasionally
            if self.frame_count % 60 == 0:
                self.print_performance_stats()

            # Update last frame time
            if hasattr(self, 'last_frame_time'):
                self.last_frame_time = time.time()
            else:
                self.last_frame_time = time.time()

    def print_performance_stats(self):
        """Print performance statistics"""
        if len(self.performance_stats['total_time']) > 0:
            avg_fps = (sum(self.fps_history) / len(self.fps_history)
                       if self.fps_history else 0)
            avg_total = np.mean(self.performance_stats['total_time']) * 1000

            active_tracks = 0
            if hasattr(self.object_detector, "tracker") and hasattr(self.object_detector.tracker, "tracks"):
                active_tracks = len(self.object_detector.tracker.tracks)

            print(f"\n[Performance] FPS: {avg_fps:.1f} | Frame Time: {avg_total:.1f}ms | "
                  f"Objects: {len(self.detections)} | Active Tracks: {active_tracks}")

    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()

        if hasattr(self.mediapipe_detector, 'release'):
            self.mediapipe_detector.release()

        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()

        print("\n" + "=" * 70)
        print("✓ Enhanced AR Vision System terminated successfully")
        print("=" * 70)


def main():
    app = None

    try:
        app = OptimizedARVisionSystem()
        app.run()

    except Exception as e:
        print("\n[ERROR] Application crashed:", e)

    finally:
        # Safe cleanup
        if app is not None:
            try:
                app.cleanup()
            except Exception as cleanup_error:
                print("[Cleanup Error]", cleanup_error)


if __name__ == "__main__":
    main()