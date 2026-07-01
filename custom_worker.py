try:
    from gunicorn.workers.geventlet import EventletWorker
except ImportError:
    # Fallback for local Windows development where Gunicorn is not installed
    class EventletWorker:
        pass

import eventlet

class CustomEventletWorker(EventletWorker):
    """
    Custom Eventlet worker for Gunicorn that only monkey patches socket and select.
    Patching thread or os breaks PyTorch C++ multithreading and Ultralytics cpuinfo module,
    causing deadlocks or crashes during YOLO model inference on cloud platforms like Render.
    """
    def patch(self):
        eventlet.monkey_patch(socket=True, select=True)
