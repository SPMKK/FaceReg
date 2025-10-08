import cv2

# --- THAY ĐỔI THÔNG TIN CỦA BẠN TẠI ĐÂY ---

# URL của luồng video từ camera IP.
# Đây là phần quan trọng nhất, mỗi loại camera có thể có một định dạng URL khác nhau.
#
# Một vài định dạng phổ biến:
# - RTSP: "rtsp://<tên_đăng_nhập>:<mật_khẩu>@<địa_chỉ_ip>:<cổng>/<đường_dẫn_luồng>"
# - HTTP: "http://<địa_chỉ_ip>:<cổng>/video" hoặc "http://<địa_chỉ_ip>/videostream.cgi"

# Ví dụ cho camera RTSP có mật khẩu:
# camera_url = "rtsp://admin:password123@192.168.1.108:554/stream1"

# Ví dụ cho camera không cần mật khẩu:
camera_url = "http://192.168.28.78:8080/14d87061586c7ce87be314ac1bf7db6e/hls/gqbb9Lhhcu/0fceca1c4aa34bd3a87853f47f841cc9/s.m3u8" # Thay thế bằng URL thực tế của bạn

# --- KẾT THÚC PHẦN THAY ĐỔI ---


# Khởi tạo đối tượng VideoCapture
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Lỗi: Không thể mở luồng video từ camera.")
    exit()

# Lấy thông số FPS (Frames Per Second) từ camera
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS của camera: {fps}")

# QUAN TRỌNG: Nhiều camera IP không trả về giá trị FPS chính xác (có thể trả về 0).
# Nếu fps = 0, chúng ta sẽ gán một giá trị mặc định, ví dụ 25 hoặc 30.
if fps == 0:
    DEFAULT_FPS = 80
    print(f"Không nhận được FPS từ camera, sử dụng giá trị mặc định: {DEFAULT_FPS}")
    fps = DEFAULT_FPS

# Tính toán thời gian delay (chờ) giữa mỗi khung hình (tính bằng mili giây)
# Công thức: delay = 1000ms / số khung hình mỗi giây
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Lỗi: Không thể nhận khung hình. Mất kết nối.")
        break

    cv2.imshow('IP Camera Stream (Real-time)', frame)

    # Thay vì waitKey(1), chúng ta đợi một khoảng 'delay' đã tính toán
    # Điều này sẽ làm cho video hiển thị đúng với tốc độ thực tế
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()