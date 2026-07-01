try:
    import eventlet
    eventlet.monkey_patch()
except ImportError:
    pass  # eventlet is only required in production (Gunicorn on Render)

import sys
import os
import time
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Ensure we can import from the BlindAssistant package
sys.path.append(os.path.join(os.path.dirname(__file__), 'BlindAssistant'))

from tracker.motion import VisionPipeline
from tracker.kalman import MovingObjectTracker
from tracker.hungarian import associate_detections_to_trackers
from tracker.voice import evaluate_and_instruct

app = Flask(__name__)
# Allow CORS for easy testing; async_mode must match the worker class
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet',
                    ping_timeout=60, ping_interval=25)

print("[INFO] Initializing Vision Pipeline (YOLO)...")
# Check inside BlindAssistant first, then project root, then let ultralytics download
project_root = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(project_root, 'BlindAssistant', 'yolov8n.pt')
if not os.path.exists(yolo_path):
    yolo_path = os.path.join(project_root, 'yolov8n.pt')
if not os.path.exists(yolo_path):
    yolo_path = "yolov8n.pt"  # let ultralytics download it
print(f"[INFO] Using YOLO model at: {yolo_path}")
vision_engine = VisionPipeline(yolo_path)

active_trackers = []
frame_width = 640

class WebTTSManager:
    """Mock TTS Manager that captures text to send to the web client."""
    def __init__(self):
        self.last_queued_time = 0
        self.current_instruction = None

    def speak(self, text):
        self.current_instruction = text
        self.last_queued_time = time.time()

# Global TTS Manager to persist 'last_queued_time' across frames
tts_manager = WebTTSManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

@socketio.on('connect')
def handle_connect():
    print("[INFO] Client connected via WebSocket")

@socketio.on('disconnect')
def handle_disconnect():
    print("[INFO] Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    global active_trackers, frame_width, tts_manager
    
    try:
        # 1. Decode incoming frame
        try:
            image_data = data['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] Frame decode failed: {e}")
            return

        if frame is None:
            return
            
        frame_width = frame.shape[1]
        
        # 2. Vision Pipeline
        detected_entities = vision_engine.process_frame(frame)
        
        # 3. Predict & Associate
        for tracker in active_trackers:
            tracker.predict()
            
        matches, unmatched_detections = associate_detections_to_trackers(detected_entities, active_trackers, iou_threshold=0.15)
        
        for d_idx, t_idx in matches:
            det_box, label = detected_entities[d_idx]
            active_trackers[t_idx].update(det_box, label)
            
        for d_idx in unmatched_detections:
            det_box, label = detected_entities[d_idx]
            active_trackers.append(MovingObjectTracker(det_box, label))
            
        # 4. Cleanup
        active_trackers = [t for t in active_trackers if t.frames_without_update < 8]
        
        # 5. Evaluate Risk & Get Instruction
        tts_manager.current_instruction = None # Reset for this frame
        evaluate_and_instruct(active_trackers, frame_width, tts_manager)
        
        # 6. Draw Bounding Boxes
        for tracker in active_trackers:
            box = tracker.get_current_box()
            x, y, w, h = box
            color = (0, 165, 255) if tracker.label == "Moving Obstacle" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, tracker.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 7. Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        data_url = 'data:image/jpeg;base64,' + encoded_image
        
        # Send back to client
        emit('processed_frame', {
            'image': data_url,
            'instruction': tts_manager.current_instruction
        })
    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        traceback.print_exc()
        emit('processed_frame', {
            'image': '',
            'instruction': f'Server error: {str(e)}'
        })

if __name__ == '__main__':
    print("[INFO] Starting Flask-SocketIO Server...")
    # Cloud providers like Render supply a PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
