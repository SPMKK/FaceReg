# src/kalman_filter.py
import cv2
import numpy as np
# ... (toàn bộ nội dung của tệp kalman_filter.py như trước) ...
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    def predict(self):
        return self.kf.predict()

    def update(self, box):
        return self.kf.correct(np.array(box, dtype=np.float32))

    def initialize(self, box):
        self.kf.statePost = np.array([box[0], box[1], box[2], box[3], 0, 0], dtype=np.float32)

def box_to_kalman(box):
    w = box[2] - box[0]; h = box[3] - box[1]; cx = box[0] + w / 2; cy = box[1] + h / 2
    return [cx, cy, w, h]

def kalman_to_box(kalman_state):
    cx, cy, w, h = kalman_state[:4]; x1 = int(cx - w / 2); y1 = int(cy - h / 2); x2 = int(cx + w / 2); y2 = int(cy + h / 2)
    return [x1, y1, x2, y2]