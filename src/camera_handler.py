# src/camera_handler.py
import cv2
import threading
import time
from collections import deque
# ... (toàn bộ nội dung của tệp camera_handler.py như trước) ...
class CameraBuffer:
    def __init__(self, src, resolution, buffer_size):
        self.src = src; self.resolution = resolution; self.buffer_size = buffer_size
        self.stopped = False; self.stream = None; self.frame_queue = deque(maxlen=buffer_size)
        self.latest_frame = None; self.lock = threading.Lock(); self.connected = False
    def _collector_thread(self):
        while not self.stopped:
            if not self.connected: self.connect(); continue
            grabbed, frame = self.stream.read()
            if not grabbed:
                print("[COLLECTOR] Mất kết nối. Đang thử kết nối lại...")
                self.stream.release(); self.connected = False; time.sleep(2.0); continue
            with self.lock: self.frame_queue.append(frame); self.latest_frame = frame
    def connect(self):
        print(f"[COLLECTOR] Đang thử kết nối tới: {self.src}...")
        self.stream = cv2.VideoCapture(self.src)
        if not self.stream.isOpened(): print("[COLLECTOR] Kết nối thất bại."); time.sleep(2.0); return
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.connected = True; print("[COLLECTOR] Kết nối thành công. Bắt đầu thu thập frame.")
    def start(self):
        thread = threading.Thread(target=self._collector_thread, args=(), daemon=True); thread.start(); return self
    def get_frame(self):
        with self.lock:
            if self.frame_queue: return self.frame_queue.popleft()
            return self.latest_frame
    def get_buffer_size(self):
        with self.lock: return len(self.frame_queue)
    def stop(self):
        self.stopped = True; time.sleep(0.5)
        if self.stream: self.stream.release()