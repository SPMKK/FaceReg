# src/config.py

# --- CẤU HÌNH BẬT/TẮT TÍNH NĂNG ---
USE_KALMAN_FILTER = True
USE_SLPT_VALIDATION = False

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_PATH = r'D:\Me\slave\run\weight\best_yolos.pt'
OPENVINO_PATH = r'D:\Me\slave\run\weight\best_yolos_openvino_model'
PROJECT_ROOT = r'D:\Me\slave\SLPT'
SLPT_MODEL_PATH = r'D:\Me\slave\WFLW_12_layer.pth'
INITIAL_POINTS_PATH = r'D:\Me\slave\SLPT\Config\init_98.npz'

# --- CẤU HÌNH CAMERA VÀ PIPELINE ---
CAMERA_INDEX = "http://192.168.28.78:8080/14d87061586c7ce87be314ac1bf7db6e/hls/gqbb9Lhhcu/0fceca1c4aa34bd3a87853f47f841cc9/s.m3u8"
CAMERA_RESOLUTION = (640, 640)
PLAYBACK_FPS = 25
BUFFER_SIZE_FRAMES = 500

# --- CẤU HÌNH MODEL VÀ TRACKING ---
DEVICE = 'cpu'
YOLO_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.3
DETECT_EVERY_N_FRAMES = 8
MAX_AGE = 30