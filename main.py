# main.py

import cv2
import time
import os
import numpy as np
import multiprocessing as mp
from queue import Empty, Full
from multiprocessing import shared_memory
from collections import defaultdict
from ultralytics import YOLO

# --- IMPORT TỪ PACKAGE 'src' ---
from Config import config
from src.camera_handler import CameraBuffer
from src.kalman_filter import KalmanFilter, box_to_kalman, kalman_to_box
from src.slpt_validator import SLPTValidator

# --------------------------------------------------------------------------
# --- CÁC HÀM CON CHO TIẾN TRÌNH AI (WORKER) ---
# --------------------------------------------------------------------------

def calculate_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa hai bounding box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


def model_inference_process(job_queue, result_queue, stop_event, worker_config):
    """
    Tiến trình riêng biệt để chạy các model AI (YOLO, SLPT).
    Hàm này chạy trên một CPU core khác.
    """
    print(f"[WORKER] Tiến trình AI (PID: {os.getpid()}) bắt đầu.")
    try:
        shm = shared_memory.SharedMemory(name=worker_config['shm_name'])
        frame_from_shm = np.ndarray(worker_config['frame_shape'], dtype=worker_config['frame_dtype'], buffer=shm.buf)
    except Exception as e:
        print(f"[WORKER-FATAL] Không thể kết nối Shared Memory: {e}")
        return

    yolo_model, slpt_validator = None, None
    try:
        print("[WORKER] Đang tải model YOLO...")
        yolo_model = YOLO(worker_config['yolo_model'])
        print("[WORKER] Tải model YOLO thành công.")
        # ... (phần tải SLPT giữ nguyên)
    except Exception as e:
        print(f"[WORKER-FATAL] Không thể tải model: {e}")
        shm.close()
        return

    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.1)
            frame_to_process = frame_from_shm.copy()
            detect_start_time = time.time()

            # --- THAY ĐỔI CHÍNH ---
            # 1. Chạy YOLO predict thay vì track
            results = yolo_model.predict(frame_to_process, imgsz=config.YOLO_IMAGE_SIZE, conf=config.CONFIDENCE_THRESHOLD, device=config.DEVICE, verbose=False)
            # Lấy các box phát hiện được, không có track_id từ YOLO nữa
            detections = results[0].boxes.xyxy.cpu().numpy()
            # --- KẾT THÚC THAY ĐỔI ---
            
            # Phần SLPT giữ nguyên logic, nhưng cách nó nhận job sẽ thay đổi ở tiến trình chính
            validation_results = {}
            debug_img_to_send = None
            if slpt_validator and job.get('validation_candidates'):
                for track_id, (x1, y1, x2, y2) in job['validation_candidates'].items():
                    face_crop = frame_to_process[y1:y2, x1:x2]
                    is_valid, debug_img = slpt_validator.validate_and_visualize(face_crop)
                    validation_results[track_id] = is_valid
                    if debug_img is not None:
                        debug_img_to_send = debug_img


            delta_time = time.time() - detect_start_time
            detection_fps = 1 / delta_time if delta_time > 0 else 0
            # Gửi về danh sách các box thô
            result_package = {'detections': detections, 'validations': validation_results, 'debug_canvas': debug_img_to_send, 'detection_fps': detection_fps}
            result_queue.put(result_package)

        except Empty:
            continue
        except Exception as e:
            print(f"[WORKER-ERROR] Lỗi: {e}")

    shm.close()
    print(f"[WORKER] Tiến trình AI (PID: {os.getpid()}) đã dừng.")

# --------------------------------------------------------------------------
# --- CÁC HÀM CON CHO VÒNG LẶP CHÍNH (PACER) ---
# --------------------------------------------------------------------------

def handle_job_submission(frame_counter, job_queue, shared_frame_np, frame, kalman_filters, validated_track_ids):
    """Gửi frame và yêu cầu xử lý cho Worker một cách định kỳ."""
    if frame_counter % config.DETECT_EVERY_N_FRAMES == 0 and job_queue.empty():
        np.copyto(shared_frame_np, frame)
        job = {}

        # **LOGIC BẬT/TẮT SLPT**
        # Chỉ tạo yêu cầu xác thực nếu SLPT được bật
        if config.USE_SLPT_VALIDATION:
            candidates = {}
            # Chỉ yêu cầu xác thực các track ID chưa được xác thực
            unvalidated_ids = set(kalman_filters.keys()) - validated_track_ids
            for track_id in unvalidated_ids:
                # Dự đoán vị trí để cắt khuôn mặt
                box = kalman_to_box(kalman_filters[track_id].predict())
                x1, y1, x2, y2 = map(int, box)
                if x1 < x2 and y1 < y2:
                    candidates[track_id] = (x1, y1, x2, y2)
            job['validation_candidates'] = candidates
        try:
            job_queue.put_nowait(job)
        except Full:
            pass # Bỏ qua nếu worker vẫn đang bận

def process_worker_results(result_queue):
    """Nhận và xử lý gói kết quả trả về từ Worker."""
    try:
        result_package = result_queue.get_nowait()
        return result_package
    except Empty:
        return None

def update_trackers_with_detections(detections, kalman_filters, track_ages, validated_track_ids, next_track_id):
    """
    Khớp các detection mới từ YOLO với các tracker hiện có bằng IoU.
    Cập nhật, tạo mới, và đánh dấu các tracker bị mất.
    """
    IOU_THRESHOLD = 0.4  # Ngưỡng để coi là một cặp khớp

    if not kalman_filters: # Nếu chưa có tracker nào, tạo mới cho tất cả detection
        for box in detections:
            kalman_filters[next_track_id].initialize(box_to_kalman(box))
            track_ages[next_track_id] = 0
            next_track_id += 1
        return next_track_id

    # Lấy vị trí dự đoán từ các tracker hiện có
    predicted_boxes = {tid: kalman_to_box(kf.predict()) for tid, kf in kalman_filters.items()}
    
    matched_indices = set()
    matched_track_ids = set()

    # Bước 1: Khớp detection với tracker
    for track_id, pred_box in predicted_boxes.items():
        best_iou = 0
        best_match_idx = -1
        for i, det_box in enumerate(detections):
            if i in matched_indices:
                continue
            iou = calculate_iou(pred_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = i
        
        if best_iou > IOU_THRESHOLD:
            # Tìm thấy một cặp khớp -> Cập nhật Kalman Filter
            kalman_filters[track_id].update(box_to_kalman(detections[best_match_idx]))
            track_ages[track_id] = 0 # Reset tuổi
            matched_indices.add(best_match_idx)
            matched_track_ids.add(track_id)

    # Bước 2: Tạo tracker mới cho các detection không khớp
    for i, det_box in enumerate(detections):
        if i not in matched_indices:
            kalman_filters[next_track_id].initialize(box_to_kalman(det_box))
            track_ages[next_track_id] = 0
            next_track_id += 1

    # Bước 3: Tăng tuổi cho các tracker không được khớp
    all_tracked_ids = list(kalman_filters.keys())
    for track_id in all_tracked_ids:
        if track_id not in matched_track_ids:
            track_ages[track_id] += 1
        # Xóa các tracker quá cũ
        if track_ages[track_id] > config.MAX_AGE:
            del kalman_filters[track_id]
            del track_ages[track_id]
            validated_track_ids.discard(track_id)
            
    return next_track_id

# Hàm này sẽ được gọi ở các frame không có detection
def predict_and_cleanup_trackers(kalman_filters, track_ages, validated_track_ids):
    """
    Dự đoán vị trí tiếp theo cho các tracker và dọn dẹp các tracker cũ.
    Hàm này không cập nhật mà chỉ predict.
    """
    all_tracked_ids = list(kalman_filters.keys())
    for track_id in all_tracked_ids:
        # Ở các frame tracking, ta không update mà chỉ predict
        # Vị trí sẽ được cập nhật bởi kf.predict() trong hàm draw_annotations
        track_ages[track_id] += 1 # Tăng tuổi vì không có detection mới
        if track_ages[track_id] > config.MAX_AGE:
            del kalman_filters[track_id]
            del track_ages[track_id]
            validated_track_ids.discard(track_id)

    # Xử lý các track bị mất và xóa các track quá cũ
    all_tracked_ids = list(kalman_filters.keys())
    for track_id in all_tracked_ids:
        if track_id not in active_ids:
            track_ages[track_id] += 1 # Tăng tuổi nếu không thấy
        if track_ages[track_id] > config.MAX_AGE:
            del kalman_filters[track_id]
            del track_ages[track_id]
            validated_track_ids.discard(track_id)

def draw_annotations(frame, latest_results, kalman_filters, track_ages, validated_track_ids):
    """Vẽ các bounding box và thông tin lên frame."""
    annotated_frame = frame.copy()
    live_boxes = {det['track_id']: det['box'] for det in latest_results['detections']}

    # **LOGIC BẬT/TẮT KALMAN**
    if config.USE_KALMAN_FILTER:
        # Vẽ dựa trên trạng thái của Kalman Filter (bao gồm cả dự đoán)
        for track_id, kf in kalman_filters.items():
            is_validated = track_id in validated_track_ids
            box, label, color = None, "", (0,0,0)

            if track_id in live_boxes: # Track đang được detect
                box = kalman_to_box(kf.predict()) # Lấy vị trí đã được làm mượt
                label = f"ID:{track_id}" if is_validated else f"ID:{track_id} (Verifying)"
                color = (0, 255, 0) if is_validated else (255, 165, 0)
            elif track_ages[track_id] > 0: # Track đã mất, dự đoán vị trí
                box = kalman_to_box(kf.predict())
                label = f"ID:{track_id} (Predicted)"
                color = (0, 255, 255)
            
            if box:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # Chỉ vẽ các box mà YOLO trả về trực tiếp
        for det in latest_results['detections']:
            track_id, box = det['track_id'], det['box']
            x1, y1, x2, y2 = map(int, box)
            label = f"ID:{track_id}"
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    return annotated_frame


# --------------------------------------------------------------------------
# --- HÀM MAIN CHÍNH ---
# --------------------------------------------------------------------------

def main():
    # --- KHỞI TẠO CÁC THÀNH PHẦN ---
    job_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()
    camera, shm, worker = None, None, None

    try:
        # --- KHỞI TẠO CAMERA VÀ BỘ ĐỆM ---
        camera = CameraBuffer(src=config.CAMERA_INDEX, resolution=config.CAMERA_RESOLUTION, buffer_size=config.BUFFER_SIZE_FRAMES).start()
        print("Đang chờ Collector lấp đầy bộ đệm ban đầu (5 giây)...")
        time.sleep(5.0)

        sample_frame = camera.get_frame()
        if sample_frame is None:
            raise RuntimeError("LỖI: Không thể lấy frame mẫu từ camera.")

        # --- KHỞI TẠO SHARED MEMORY VÀ TIẾN TRÌNH WORKER ---
        frame_shape, frame_dtype, frame_size_bytes = sample_frame.shape, sample_frame.dtype, sample_frame.nbytes
        shm = shared_memory.SharedMemory(create=True, size=frame_size_bytes)
        shared_frame_np = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)

        worker_config = {
            'yolo_model': config.OPENVINO_PATH,
            'use_slpt': config.USE_SLPT_VALIDATION,
            'shm_name': shm.name,
            'frame_shape': frame_shape,
            'frame_dtype': frame_dtype
        }
        worker = mp.Process(target=model_inference_process, args=(job_queue, result_queue, stop_event, worker_config), daemon=True)
        worker.start()

        # --- KHỞI TẠO CÁC BIẾN TRẠNG THÁI ---
        target_frame_time = 1.0 / config.PLAYBACK_FPS
        frame_counter, detection_fps = 0, 0.0
        kalman_filters = defaultdict(KalmanFilter)
        track_ages = defaultdict(int)
        validated_track_ids = set()
        debug_canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Biến mới để quản lý ID cho tracker
        next_track_id = 1

        print(f"Bắt đầu Pacer với tốc độ mục tiêu: {config.PLAYBACK_FPS} FPS.")
        print(f"Logic: Detect mỗi {config.DETECT_EVERY_N_FRAMES} frames, còn lại tracking bằng Kalman Filter.")
        print(f"Kalman Filter: {'BẬT' if config.USE_KALMAN_FILTER else 'TẮT'}")
        print(f"SLPT Validation: {'BẬT' if config.USE_SLPT_VALIDATION else 'TẮT'}")

        # --- VÒNG LẶP CHÍNH ---
        while True:
            loop_start_time = time.time()
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01) # Chờ frame mới
                continue

            # Xác định xem frame hiện tại có phải là frame để chạy detection không
            is_detect_frame = (frame_counter % config.DETECT_EVERY_N_FRAMES == 0)

            # 1. Gửi yêu cầu xử lý cho Worker nếu đến lượt
            if is_detect_frame:
                handle_job_submission(frame_counter, job_queue, shared_frame_np, frame, kalman_filters, validated_track_ids)

            # 2. Nhận kết quả từ Worker (nếu có) và cập nhật tracker
            result_package = process_worker_results(result_queue)
            if result_package:
                # Có kết quả detection mới -> thực hiện data association
                new_detections = result_package['detections']
                detection_fps = result_package.get('detection_fps', detection_fps)
                
                if config.USE_KALMAN_FILTER:
                    next_track_id = update_trackers_with_detections(
                        new_detections, kalman_filters, track_ages, validated_track_ids, next_track_id
                    )
                
                # Cập nhật các thông tin phụ (SLPT, debug canvas)
                if result_package.get('debug_canvas') is not None:
                    debug_canvas = result_package['debug_canvas']
                if config.USE_SLPT_VALIDATION:
                    for track_id, is_valid in result_package.get('validations', {}).items():
                        if is_valid:
                            validated_track_ids.add(track_id)
            
            # 3. Nếu không phải frame detect và không có kết quả mới -> chỉ predict và dọn dẹp
            elif not is_detect_frame and config.USE_KALMAN_FILTER:
                predict_and_cleanup_trackers(kalman_filters, track_ages, validated_track_ids)

            # 4. Vẽ kết quả lên frame
            # Hàm draw_annotations sẽ vẽ dựa trên trạng thái hiện tại của kalman_filters
            # (bao gồm cả các box được dự đoán ở frame tracking)
            annotated_frame = draw_annotations(frame, {'detections': []}, kalman_filters, track_ages, validated_track_ids)

            # 5. Hiển thị thông tin và frame
            buffer_fill = camera.get_buffer_size()
            cv2.putText(annotated_frame, f"Buffer: {buffer_fill}/{config.BUFFER_SIZE_FRAMES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detect FPS: {int(detection_fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Tracking", annotated_frame)
            if config.USE_SLPT_VALIDATION:
                cv2.imshow("SLPT Validation Debug", debug_canvas)

            # 6. Điều tốc vòng lặp và xử lý thoát
            frame_counter += 1
            processing_time = time.time() - loop_start_time
            sleep_time = target_frame_time - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Đã xảy ra lỗi nghiêm trọng trong hàm main: {e}")
    finally:
        # --- DỌN DẸP TÀI NGUYÊN ---
        print("Đang dừng các tiến trình và dọn dẹp...")
        stop_event.set()
        if worker and worker.is_alive():
            worker.join(timeout=2)
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        if shm:
            shm.close()
            shm.unlink()
            print("Đã dọn dẹp Shared Memory.")
        print("Đã thoát.")


if __name__ == "__main__":
    mp.freeze_support()
    main()