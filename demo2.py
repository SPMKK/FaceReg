# Thêm các import cần thiết cho multiprocessing
import multiprocessing as mp
from queue import Empty # Dùng để bắt lỗi khi hàng đợi rỗng

# --- PHẦN 1, 2, 3: Cấu hình, Import, và các lớp tiện ích (Giữ nguyên) ---
# ... (Toàn bộ code từ Phần 1 đến 3 của bạn không thay đổi) ...
# Ví dụ một vài phần:
import cv2
import time
import threading
import numpy as np
import sys
import os
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from collections import defaultdict

# --- CẤU HÌNH YOLO ---
MODEL_PATH = r'D:\Me\slave\run\weight\best_yolos.pt'
OPENVINO = r'D:\Me\slave\run\weight\best_yolos_openvino_model'
DEVICE = 'cpu'
# --- CẤU HÌNH SLPT ---
PROJECT_ROOT = r'D:\Me\slave\SLPT'
SLPT_MODEL_PATH = r'D:\Me\slave\WFLW_12_layer.pth'
# --- CẤU HÌNH CHUNG ---
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (640, 640)
DETECT_EVERY_N_FRAMES = 2 # Giảm N để pipeline luôn có việc làm
MAX_AGE = 30 # Giảm MAX_AGE để dọn dẹp track nhanh hơn
YOLO_IMAGE_SIZE = 320
CONFIDENCE_THRESHOLD = 0.7

sys.path.append(PROJECT_ROOT)
try:
    from Config import cfg
    from SLPT.SLPT import Sparse_alignment_network
except ImportError as e:
    print(f"LỖI: Không thể import các module từ dự án SLPT: {e}")
    sys.exit(1)

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    def predict(self): return self.kf.predict()
    def update(self, box): return self.kf.correct(np.array(box, dtype=np.float32))
    def initialize(self, box): self.kf.statePost = np.array([box[0], box[1], box[2], box[3], 0, 0], dtype=np.float32)

class CameraStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"LỖI: Không thể mở camera index {src}")
            self.stopped = True
            return
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self): threading.Thread(target=self.update, args=(), daemon=True).start(); return self
    def update(self):
        while not self.stopped: (self.grabbed, self.frame) = self.stream.read()
        self.stream.release()
    def read(self): return self.frame
    def stop(self): self.stopped = True

def box_to_kalman(box):
    w = box[2] - box[0]; h = box[3] - box[1]
    cx = box[0] + w / 2; cy = box[1] + h / 2
    return [cx, cy, w, h]

def kalman_to_box(kalman_state):
    cx, cy, w, h = kalman_state[:4]
    x1 = int(cx - w / 2); y1 = int(cy - h / 2)
    x2 = int(cx + w / 2); y2 = int(cy + h / 2)
    return [x1, y1, x2, y2]

def validate_and_visualize_face(face_crop, slpt_model, cfg, transform):
    # ... (Code của bạn giữ nguyên) ...
    if face_crop.size == 0:
        return False, np.zeros((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE, 3), dtype=np.uint8)
    validation_result = False
    debug_image = cv2.resize(face_crop, (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))
    try:
        face_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0)
        with torch.no_grad():
            landmarks_list = slpt_model(face_tensor)
        predicted_landmarks = landmarks_list[2][0, -1].cpu().numpy()
        for point in predicted_landmarks:
            cv2.circle(debug_image, (int(point[0] * cfg.MODEL.IMG_SIZE), int(point[1] * cfg.MODEL.IMG_SIZE)), 1, (0, 255, 255), -1)
        left_eye = predicted_landmarks[60]
        right_eye = predicted_landmarks[72]
        eye_dist_normalized = np.linalg.norm(left_eye - right_eye)
        eye_dist_pixels = eye_dist_normalized * cfg.MODEL.IMG_SIZE
        if eye_dist_pixels > 30:
            validation_result = True
    except Exception as e:
        # Quan trọng: print lỗi ra để debug
        print(f"[SLPT-WORKER-ERROR] Lỗi xác thực khuôn mặt: {e}")
        validation_result = False
    return validation_result, debug_image


# --- PHẦN 4: HÀM XỬ LÝ MODEL CHO TIẾN TRÌNH RIÊNG (WORKER PROCESS) ---
def model_inference_process(job_queue, result_queue, stop_event, model_paths_config):
    """
    Hàm này chạy trong một tiến trình hoàn toàn riêng biệt.
    Nó tải model một lần và sau đó đi vào vòng lặp xử lý.
    """
    print(f"[WORKER] Tiến trình xử lý (PID: {os.getpid()}) bắt đầu.")
    try:
        # Tải models bên trong tiến trình con
        yolo_model = YOLO(model_paths_config['yolo'])
        
        # Thêm lại sys.path vì tiến trình con không kế thừa nó
        sys.path.append(model_paths_config['project_root'])
        from Config import cfg as worker_cfg
        from SLPT.SLPT import Sparse_alignment_network

        initial_points_path = os.path.join(model_paths_config['project_root'], 'Config', 'init_98.npz')
        worker_cfg.TRANSFORMER.NUM_DECODER = 12
        slpt_model = Sparse_alignment_network(
            num_point=worker_cfg.WFLW.NUM_POINT, d_model=worker_cfg.MODEL.OUT_DIM, trainable=False,
            return_interm_layers=True, dilation=worker_cfg.MODEL.DILATION, nhead=worker_cfg.TRANSFORMER.NHEAD,
            feedforward_dim=worker_cfg.TRANSFORMER.FEED_DIM, initial_path=initial_points_path, cfg=worker_cfg
        )
        checkpoint = torch.load(model_paths_config['slpt'], map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        slpt_model.load_state_dict(new_state_dict)
        slpt_model.eval()
        slpt_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print("[WORKER] Tất cả model đã được tải thành công.")
    except Exception as e:
        print(f"[WORKER-FATAL] Không thể tải model. Tiến trình sẽ dừng. Lỗi: {e}")
        return

    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.1)
            frame_to_process, validation_candidates = job
            
            detect_start_time = time.time()
            
            # 1. Chạy YOLO Tracking
            results = yolo_model.track(frame_to_process, persist=True, imgsz=YOLO_IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, device=DEVICE, verbose=False)
            
            detection_results = []
            if results[0].boxes.id is not None:
                all_boxes = results[0].boxes.xyxy.cpu().numpy()
                all_track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(all_boxes, all_track_ids):
                    detection_results.append({'box': box, 'track_id': track_id})

            # 2. Chạy SLPT Validation cho các ứng viên mới
            validation_results = {}
            debug_img_to_send = None
            for track_id, (x1, y1, x2, y2) in validation_candidates.items():
                face_crop = frame_to_process[y1:y2, x1:x2]
                is_valid, debug_img = validate_and_visualize_face(face_crop, slpt_model, worker_cfg, slpt_transform)
                validation_results[track_id] = is_valid
                if debug_img is not None:
                    debug_img_to_send = debug_img # Chỉ gửi ảnh debug cuối cùng

            detect_end_time = time.time()
            detection_fps = 1 / (detect_end_time - detect_start_time) if (detect_end_time - detect_start_time) > 0 else 0

            # 3. Gửi kết quả về
            result_package = {
                'detections': detection_results,
                'validations': validation_results,
                'debug_canvas': debug_img_to_send,
                'detection_fps': detection_fps
            }
            result_queue.put(result_package)

        except Empty:
            continue # Hàng đợi rỗng, tiếp tục vòng lặp để kiểm tra stop_event
        except Exception as e:
            print(f"[WORKER-ERROR] Đã xảy ra lỗi trong vòng lặp xử lý: {e}")

    print(f"[WORKER] Tiến trình xử lý (PID: {os.getpid()}) đã dừng.")


# --- PHẦN 5: HÀM MAIN CHÍNH (ĐÃ CẬP NHẬT HOÀN TOÀN) ---
def main():
    # --- KHỞI TẠO CÁC HÀNG ĐỢI VÀ SỰ KIỆN DỪNG ---
    job_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    # --- KHỞI TẠO VÀ BẮT ĐẦU TIẾN TRÌNH WORKER ---
    model_paths_config = {'yolo': OPENVINO, 'slpt': SLPT_MODEL_PATH, 'project_root': PROJECT_ROOT}
    worker = mp.Process(
        target=model_inference_process,
        args=(job_queue, result_queue, stop_event, model_paths_config),
        daemon=True
    )
    worker.start()

    # --- KHỞI TẠO CÁC BIẾN CHO TIẾN TRÌNH CHÍNH ---
    camera = CameraStream(src=CAMERA_INDEX, resolution=CAMERA_RESOLUTION).start()
    if camera.stopped:
        stop_event.set(); worker.join(); return
        
    frame_counter = 0; prev_frame_time = 0; smoothed_fps = 0.0; alpha = 0.1; detection_fps = 0.0
    
    kalman_filters = {}; track_ages = {}; validated_track_ids = set(); unvalidated_track_ids = set()
    
    debug_canvas = np.zeros((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE, 3), dtype=np.uint8)
    latest_detections = [] 

    # --- VÒNG LẶP CHÍNH ---
    while True:
        frame = camera.read()
        if frame is None: continue

        frame_counter += 1
        annotated_frame = frame.copy()

        # 1. Gửi việc nếu đến lúc và worker đang rảnh
        if frame_counter % DETECT_EVERY_N_FRAMES == 0:
            if job_queue.empty(): 
                validation_candidates = {}
                for track_id in unvalidated_track_ids:
                    for det in latest_detections:
                        if det['track_id'] == track_id:
                            x1, y1, x2, y2 = map(int, det['box'])
                            validation_candidates[track_id] = (x1, y1, x2, y2)
                            break
                
                job_package = (frame, validation_candidates)
                job_queue.put(job_package)

        # 2. Nhận kết quả nếu có
        try:
            result_package = result_queue.get_nowait()
            latest_detections = result_package['detections']
            detection_fps = result_package['detection_fps']
            if result_package['debug_canvas'] is not None:
                debug_canvas = result_package['debug_canvas']
            
            for track_id, is_valid in result_package['validations'].items():
                if is_valid:
                    validated_track_ids.add(track_id)
                unvalidated_track_ids.discard(track_id)

        except Empty:
            pass 

        # 3. Cập nhật Kalman dựa trên kết quả YOLO mới nhất
        active_track_ids_in_frame = []
        for det in latest_detections:
            track_id = det['track_id']
            box = det['box']
            active_track_ids_in_frame.append(track_id)
            kalman_measurement = box_to_kalman(box)
            
            if track_id not in kalman_filters:
                kalman_filters[track_id] = KalmanFilter()
                kalman_filters[track_id].initialize(kalman_measurement)
                if track_id not in validated_track_ids:
                    unvalidated_track_ids.add(track_id)
            else:
                kalman_filters[track_id].update(kalman_measurement)
            track_ages[track_id] = 0

        # Tăng tuổi cho tất cả các track, tuổi sẽ được reset về 0 ở trên nếu track được phát hiện
        for track_id in kalman_filters:
            track_ages[track_id] = track_ages.get(track_id, 0) + 1

        # --- THAY ĐỔI LOGIC TẠI ĐÂY ---
        # 4. Dự đoán và vẽ tất cả các track đang hoạt động
        for track_id, kf in list(kalman_filters.items()):
            is_validated = track_id in validated_track_ids
            is_active = track_id in active_track_ids_in_frame

            if is_active:
                # Track đang được YOLO phát hiện trong frame hiện tại
                predicted_state = kf.kf.statePost
                if is_validated:
                    # Màu xanh lá: Đã xác thực
                    label, color = f"ID: {track_id}", (0, 255, 0)
                else:
                    # Màu cam: Mới phát hiện, đang chờ xác thực
                    label, color = f"ID: {track_id} (Pending)", (255, 165, 0) 
            else:
                # Track không được phát hiện, dùng Kalman để dự đoán vị trí
                # Chỉ hiển thị dự đoán cho những track ĐÃ được xác thực để tránh nhiễu
                if is_validated:
                    predicted_state = kf.predict()
                    # Màu vàng: Dự đoán vị trí của track đã xác thực
                    label, color = f"ID: {track_id} (Predicted)", (0, 255, 255)
                else:
                    # Bỏ qua, không vẽ những track chưa xác thực đã bị mất dấu
                    continue
            
            # Vẽ bounding box và label lên frame
            tracked_box = kalman_to_box(predicted_state)
            x1, y1, x2, y2 = tracked_box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # --- KẾT THÚC THAY ĐỔI ---

        # 5. Dọn dẹp các track cũ
        for track_id in list(kalman_filters.keys()):
            if track_ages.get(track_id, 0) > MAX_AGE:
                del kalman_filters[track_id]; del track_ages[track_id]
                validated_track_ids.discard(track_id); unvalidated_track_ids.discard(track_id)

        # Hiển thị thông tin
        new_frame_time = time.time(); instant_fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        smoothed_fps = (alpha * instant_fps) + (1.0 - alpha) * smoothed_fps; prev_frame_time = new_frame_time
        cv2.putText(annotated_frame, f"App FPS: {int(smoothed_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Detect FPS: {int(detection_fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Face Tracking", annotated_frame); cv2.imshow("SLPT Validation Debug", debug_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    # Dọn dẹp
    print("Đang dừng các tiến trình...")
    stop_event.set()
    worker.join(timeout=2) # Thêm timeout để tránh bị treo
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Bắt buộc cho multiprocessing trên Windows
    mp.freeze_support()
    main()