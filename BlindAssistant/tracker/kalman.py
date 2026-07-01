import numpy as np
from filterpy.kalman import KalmanFilter

class MovingObjectTracker:
    """
    A Kalman Filter based tracker for maintaining the state (position, velocity, size)
    of a bounding box across multiple frames.
    """
    def __init__(self, bbox, label):
        # 7 states: [center_x, center_y, area, aspect_ratio, vel_x, vel_y, vel_area]
        # 4 measurements: [center_x, center_y, area, aspect_ratio]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State Transition Matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0], # x_{t} = x_{t-1} + vx_{t-1}
            [0, 1, 0, 0, 0, 1, 0], # y_{t} = y_{t-1} + vy_{t-1}
            [0, 0, 1, 0, 0, 0, 1], # area_{t} = area_{t-1} + v_area_{t-1}
            [0, 0, 0, 1, 0, 0, 0], # aspect_{t} = aspect_{t-1}
            [0, 0, 0, 0, 1, 0, 0], # vx_{t} = vx_{t-1}
            [0, 0, 0, 0, 0, 1, 0], # vy_{t} = vy_{t-1}
            [0, 0, 0, 0, 0, 0, 1]  # v_area_{t} = v_area_{t-1}
        ])
        
        # Measurement Matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Initialize state with the first bounding box observation
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        area = w * h
        aspect = w / float(h) if h > 0 else 1
        
        self.kf.x[:4] = np.array([cx, cy, area, aspect]).reshape((4, 1))
        
        self.label = label
        self.frames_without_update = 0
        self.last_z = None
        self.vz_mps_smoothed = 0.0

    def predict(self):
        """
        Predicts the next state of the bounding box.
        """
        self.kf.predict()
        self.frames_without_update += 1
        return self.get_current_box()

    def update(self, bbox, label):
        """
        Updates the state with a new bounding box observation.
        """
        self.frames_without_update = 0
        # Potentially update label if YOLO got a better read (e.g. from Unknown to Person)
        self.label = label 
        
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        area = w * h
        aspect = w / float(h) if h > 0 else 1
        
        z = np.array([cx, cy, area, aspect]).reshape((4, 1))
        self.kf.update(z)

    def get_current_box(self):
        """
        Converts internal Kalman state back into [x, y, w, h] bounding box.
        """
        cx, cy = self.kf.x[0, 0], self.kf.x[1, 0]
        area = self.kf.x[2, 0]
        aspect = self.kf.x[3, 0]
        
        if area <= 0 or aspect <= 0:
            return [0, 0, 1, 1]
            
        w = np.sqrt(area * aspect)
        h = area / w
        return [int(cx - w/2), int(cy - h/2), int(w), int(h)]

    def get_velocity_mps(self, focal_length, real_width, fps):
        """
        Extracts the velocity from the Kalman state and converts it from pixels/frame to meters/second.
        Returns: (velocity_x_mps, velocity_z_mps)
        """
        if self.frames_without_update > 0:
            return 0.0, 0.0 # Don't trust velocity if we are just extrapolating
            
        # State: [cx, cy, area, aspect, vx, vy, v_area]
        vx_pixels_per_frame = self.kf.x[4, 0]
        v_area_per_frame = self.kf.x[6, 0]
        
        box = self.get_current_box()
        w = box[2]
        
        if w <= 0:
            return 0.0, 0.0
            
        # Current distance Z
        current_z = (real_width * focal_length) / w
        
        # X velocity (lateral): (vx_pixels * Z) / f
        vx_mps = (vx_pixels_per_frame * current_z / focal_length) * fps
        
        # Z velocity (approaching/leaving): calculate rate of change of depth
        if self.last_z is not None:
            # Positive vz means approaching (distance Z is decreasing over time)
            raw_vz = (self.last_z - current_z) * fps
            self.vz_mps_smoothed = 0.4 * self.vz_mps_smoothed + 0.6 * raw_vz
        else:
            # Estimate from area rate of change if we don't have previous depth yet
            if v_area_per_frame > 20:
                self.vz_mps_smoothed = 0.8
            elif v_area_per_frame < -20:
                self.vz_mps_smoothed = -0.8
            else:
                self.vz_mps_smoothed = 0.0
                
        self.last_z = current_z
        return round(float(vx_mps), 2), round(float(self.vz_mps_smoothed), 2)
