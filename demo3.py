import cv2
import time
import threading
import numpy as np
import sys
import os
import torch
import torchvision.transforms.v2 as transforms
from ultralytics import YOLO
from collections import defaultdict, deque
import multiprocessing as mp
from queue import Empty, Full
from multiprocessing import shared_memory

# --- CẤU HÌNH YOLO ---
MODEL_PATH = r'D:\Me\slave\run\weight\best_yolos.pt'
OPENVINO = r'D:\Me\slave\run\weight\best_yolos_openvino_model'
DEVICE = 'cpu'
# --- CẤU HÌNH SLPT ---
PROJECT_ROOT = r'D:\Me\slave\SLPT'
SLPT_MODEL_PATH = r'D:\Me\slave\WFLW_12_layer.pth'
# --- CẤU HÌNH CHUNG ---
# --- THAY ĐỔI: Điền URL camera của bạn vào đây ---
# Ví dụ: "rtsp://admin:password@192.168.1.108/stream1" hoặc 0 cho webcam
CAMERA_INDEX = "http://192.168.28.78:8080/14d87061586c7ce87be314ac1bf7db6e/hls/gqbb9Lhhcu/0fceca1c4aa34bd3a87853f47f841cc9/s.m3u8" 
CAMERA_RESOLUTION = (640, 640)
DETECT_EVERY_N_FRAMES = 8
MAX_AGE = 30
YOLO_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.3
# --- MỚI: CẤU HÌNH PIPELINE VÀ BỘ ĐỆM ---
# --------------------------------------------------------------------------
# Tốc độ FPS mà bạn muốn ứng dụng hiển thị một cách mượt mà.
# Đây là tốc độ của "Giai đoạn 2: The Pacer".
PLAYBACK_FPS = 25

# Kích thước của bộ đệm (Buffer Queue) tính bằng số lượng frame.
# Phải đủ lớn để chứa được "cục" frame lớn nhất từ camera.
# Ví dụ: nếu camera là 30fps và delay 8s, cục frame có thể lên tới 240.
# Chọn một giá trị an toàn như 500.
BUFFER_SIZE_FRAMES = 500
# --------------------------------------------------------------------------
sys.path.append(PROJECT_ROOT)
try:
    from Config import cfg
    from SLPT.SLPT import Sparse_alignment_network
except ImportError as e:
    print(f"LỖI: Không thể import các module từ dự án SLPT: {e}")
    sys.exit(1)

class KalmanFilter: # (Không thay đổi)
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    def predict(self): return self.kf.predict()
    def update(self, box): return self.kf.correct(np.array(box, dtype=np.float32))
    def initialize(self, box): self.kf.statePost = np.array([box[0], box[1], box[2], box[3], 0, 0], dtype=np.float32)

class CameraBuffer: # (Giai đoạn 1: The Collector - Giữ nguyên)
    def __init__(self, src, resolution, buffer_size):
        self.src = src
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.stopped = False
        self.stream = None
        self.frame_queue = deque(maxlen=buffer_size)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.connected = False
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
            if self.frame_queue: frame = self.frame_queue.popleft(); self.latest_frame = frame; return frame
            return self.latest_frame
    def get_buffer_size(self):
        with self.lock: return len(self.frame_queue)
    def stop(self):
        self.stopped = True; time.sleep(0.5); 
        if self.stream: self.stream.release()

def box_to_kalman(box):
    w = box[2] - box[0]; h = box[3] - box[1]; cx = box[0] + w / 2; cy = box[1] + h / 2
    return [cx, cy, w, h]

def kalman_to_box(kalman_state):
    cx, cy, w, h = kalman_state[:4]; x1 = int(cx - w / 2); y1 = int(cy - h / 2); x2 = int(cx + w / 2); y2 = int(cy + h / 2)
    return [x1, y1, x2, y2]

def validate_and_visualize_face(face_crop, slpt_model, cfg, transform):
    if face_crop.size == 0: return False, np.zeros((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE, 3), dtype=np.uint8)
    validation_result = False; debug_image = cv2.resize(face_crop, (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))
    try:
        face_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB); face_tensor = transform(face_rgb).unsqueeze(0)
        with torch.no_grad(): landmarks_list = slpt_model(face_tensor)
        predicted_landmarks = landmarks_list[2][0, -1].cpu().numpy()
        for point in predicted_landmarks: cv2.circle(debug_image, (int(point[0] * cfg.MODEL.IMG_SIZE), int(point[1] * cfg.MODEL.IMG_SIZE)), 1, (0, 255, 255), -1)
        left_eye, right_eye = predicted_landmarks[60], predicted_landmarks[72]
        eye_dist_pixels = np.linalg.norm(left_eye - right_eye) * cfg.MODEL.IMG_SIZE
        if eye_dist_pixels > 30: validation_result = True
    except Exception as e: print(f"[SLPT-WORKER-ERROR] Lỗi: {e}"); validation_result = False
    return validation_result, debug_image

# --- Giai đoạn 3: The Thinker (Tiến trình xử lý AI) ---
def model_inference_process(job_queue, result_queue, stop_event, config):
    print(f"[WORKER] Tiến trình AI (PID: {os.getpid()}) bắt đầu.")
    try:
        existing_shm = shared_memory.SharedMemory(name=config['shm_name'])
        frame_from_shm = np.ndarray(config['frame_shape'], dtype=config['frame_dtype'], buffer=existing_shm.buf)
    except Exception as e: print(f"[WORKER-FATAL] Không thể kết nối Shared Memory: {e}"); return
    
    try:
        # --- TẢI MODEL YOLO TẠI ĐÂY (TRONG TIẾN TRÌNH RIÊNG) ---
        print("[WORKER] Đang tải model YOLO...")
        yolo_model = YOLO(config['yolo'])
        print("[WORKER] Tải model YOLO thành công.")

        # --- TẢI MODEL SLPT TẠI ĐÂY (TRONG TIẾN TRÌNH RIÊNG) ---
        print("[WORKER] Đang tải model SLPT...")
        sys.path.append(config['project_root'])
        from Config import cfg as worker_cfg
        from SLPT.SLPT import Sparse_alignment_network
        initial_points_path = os.path.join(config['project_root'], 'Config', 'init_98.npz')
        worker_cfg.TRANSFORMER.NUM_DECODER = 12
        slpt_model = Sparse_alignment_network(num_point=worker_cfg.WFLW.NUM_POINT, d_model=worker_cfg.MODEL.OUT_DIM, trainable=False, return_interm_layers=True, dilation=worker_cfg.MODEL.DILATION, nhead=worker_cfg.TRANSFORMER.NHEAD, feedforward_dim=worker_cfg.TRANSFORMER.FEED_DIM, initial_path=initial_points_path, cfg=worker_cfg)
        checkpoint = torch.load(config['slpt'], map_location='cpu')
        slpt_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
        slpt_model.eval()
        slpt_transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print("[WORKER] Tải model SLPT thành công.")
    except Exception as e: print(f"[WORKER-FATAL] Không thể tải model: {e}"); existing_shm.close(); return

    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.1)
            frame_to_process = frame_from_shm.copy()
            validation_candidates = job.get('validation_candidates', {})
            detect_start_time = time.time()

            # --- THỰC THI YOLO TRACKING TẠI ĐÂY (TRONG TIẾN TRÌNH RIÊNG) ---
            results = yolo_model.track(frame_to_process, persist=True, imgsz=YOLO_IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, device=DEVICE, verbose=False)
            
            detection_results = [{'box': box, 'track_id': track_id} for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy().astype(int))] if results[0].boxes.id is not None else []
            validation_results = {}
            debug_img_to_send = None
            if validation_candidates:
                for track_id, (x1, y1, x2, y2) in validation_candidates.items():
                    face_crop = frame_to_process[y1:y2, x1:x2]
                    is_valid, debug_img = validate_and_visualize_face(face_crop, slpt_model, worker_cfg, slpt_transform)
                    validation_results[track_id] = is_valid
                    if debug_img is not None: debug_img_to_send = debug_img
            delta_time = time.time() - detect_start_time
            detection_fps = 1 / delta_time if delta_time > 0 else 0
            result_package = {'detections': detection_results, 'validations': validation_results, 'debug_canvas': debug_img_to_send, 'detection_fps': detection_fps}
            result_queue.put(result_package)
        except Empty: continue
        except Exception as e: print(f"[WORKER-ERROR] Lỗi: {e}")
    existing_shm.close()
    print(f"[WORKER] Tiến trình AI (PID: {os.getpid()}) đã dừng.")

def main():
    job_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()
    camera = None
    shm = None
    worker = None

    try:
        # Giai đoạn 1: Khởi tạo Collector
        camera = CameraBuffer(src=CAMERA_INDEX, resolution=CAMERA_RESOLUTION, buffer_size=BUFFER_SIZE_FRAMES).start()
        print("Đang chờ Collector lấp đầy bộ đệm ban đầu (5 giây)...")
        time.sleep(5.0)
        
        sample_frame = camera.get_frame()
        if sample_frame is None:
            raise RuntimeError("LỖI: Không thể lấy frame mẫu từ camera. Kiểm tra URL/kết nối.")

        # Thiết lập Shared Memory để giao tiếp giữa Pacer và Thinker
        frame_shape = sample_frame.shape; frame_dtype = sample_frame.dtype; frame_size_bytes = sample_frame.nbytes
        shm = shared_memory.SharedMemory(create=True, size=frame_size_bytes)
        shared_frame_np = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)

        # Giai đoạn 3: Khởi tạo Thinker
        config = {'yolo': OPENVINO, 'slpt': SLPT_MODEL_PATH, 'project_root': PROJECT_ROOT, 'shm_name': shm.name, 'frame_shape': frame_shape, 'frame_dtype': frame_dtype}
        worker = mp.Process(target=model_inference_process, args=(job_queue, result_queue, stop_event, config), daemon=True)
        worker.start()

        # Giai đoạn 2: Bắt đầu Pacer (Vòng lặp chính)
        target_frame_time = 1.0 / PLAYBACK_FPS
        print(f"Bắt đầu Pacer với tốc độ mục tiêu: {PLAYBACK_FPS} FPS.")

        frame_counter = 0; detection_fps = 0.0
        kalman_filters = defaultdict(KalmanFilter); track_ages = defaultdict(int); validated_track_ids = set()
        debug_canvas = np.zeros((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE, 3), dtype=np.uint8)
        latest_results = {'detections': [], 'validations': {}}

        while True:
            loop_start_time = time.time()
            frame = camera.get_frame()
            if frame is None: time.sleep(0.1); continue
            
            annotated_frame = frame.copy()

            if frame_counter % DETECT_EVERY_N_FRAMES == 0:
                if job_queue.empty():
                    np.copyto(shared_frame_np, frame)
                    validation_candidates = {}
                    unvalidated_track_ids = set(kalman_filters.keys()) - validated_track_ids
                    for track_id in unvalidated_track_ids:
                        box = kalman_to_box(kalman_filters[track_id].predict())
                        x1, y1, x2, y2 = map(int, box)
                        if x1 < x2 and y1 < y2: validation_candidates[track_id] = (x1, y1, x2, y2)
                    try: job_queue.put_nowait({'validation_candidates': validation_candidates})
                    except Full: pass

            try:
                result_package = result_queue.get_nowait()
                latest_results = result_package
                detection_fps = result_package.get('detection_fps', detection_fps)
                if result_package.get('debug_canvas') is not None: debug_canvas = result_package['debug_canvas']
                for track_id, is_valid in result_package.get('validations', {}).items():
                    if is_valid: validated_track_ids.add(track_id)
            except Empty: pass

            active_track_ids_in_frame = {det['track_id'] for det in latest_results.get('detections', [])}
            if 'detections' in latest_results:
                for det in latest_results['detections']:
                    track_id, box = det['track_id'], det['box']
                    kalman_measurement = box_to_kalman(box)
                    if track_id not in kalman_filters: kalman_filters[track_id].initialize(kalman_measurement)
                    else: kalman_filters[track_id].update(kalman_measurement)
                    track_ages[track_id] = 0
            all_tracked_ids = list(kalman_filters.keys())
            for track_id in all_tracked_ids:
                if track_id not in active_track_ids_in_frame: track_ages[track_id] += 1
                if track_ages[track_id] > MAX_AGE:
                    del kalman_filters[track_id]; del track_ages[track_id]
                    validated_track_ids.discard(track_id)
            live_boxes = {det['track_id']: det['box'] for det in latest_results.get('detections', [])}
            for track_id, kf in kalman_filters.items():
                is_validated = track_id in validated_track_ids; box = None
                if track_id in live_boxes:
                    box = live_boxes[track_id]
                    label, color = (f"ID: {track_id}", (0, 255, 0)) if is_validated else (f"ID: {track_id} (Pending)", (255, 165, 0))
                elif track_ages[track_id] == 0 or is_validated:
                    box = kalman_to_box(kf.predict())
                    label, color = f"ID: {track_id} (Predicted)", (0, 255, 255)
                if box:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            buffer_fill = camera.get_buffer_size()
            cv2.putText(annotated_frame, f"Buffer: {buffer_fill}/{BUFFER_SIZE_FRAMES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detect FPS: {int(detection_fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Tracking", annotated_frame)
            cv2.imshow("SLPT Validation Debug", debug_canvas)
            frame_counter += 1
            
            processing_time = time.time() - loop_start_time
            sleep_time = target_frame_time - processing_time
            if sleep_time > 0: time.sleep(sleep_time)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    except Exception as e:
        print(f"Đã xảy ra lỗi nghiêm trọng trong hàm main: {e}")
    finally:
        # --- DỌN DẸP TÀI NGUYÊN ---
        print("Đang dừng các tiến trình và dọn dẹp...")
        stop_event.set()
        if worker and worker.is_alive(): worker.join(timeout=2)
        if camera: camera.stop()
        cv2.destroyAllWindows()
        if shm: shm.close(); shm.unlink(); print("Đã dọn dẹp Shared Memory.")
        print("Đã thoát.")

if __name__ == "__main__":
    mp.freeze_support()
    main()

