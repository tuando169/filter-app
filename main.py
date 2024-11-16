import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Đọc file kính râm với alpha channel (PNG)
sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

# Kiểm tra nếu file không được đọc
if sunglasses is None:
    print("Không thể đọc file sunglasses.png. Vui lòng kiểm tra lại đường dẫn hoặc nội dung file.")
    exit()

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ảnh theo trục dọc (như gương)
    frame = cv2.flip(frame, 1)

    # Chuyển đổi sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Lấy kích thước khung hình
            h, w, _ = frame.shape

            # Lấy tọa độ mắt trái, mắt phải và mũi
            left_eye = face_landmarks.landmark[33]  # Mắt trái
            right_eye = face_landmarks.landmark[263]  # Mắt phải

            # Đảo vị trí mắt trái và phải do ảnh bị lật
            left_eye, right_eye = right_eye, left_eye

            # Chuyển đổi tọa độ từ tỷ lệ sang pixel
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            # Tính khoảng cách giữa hai mắt (để điều chỉnh kích thước kính)
            eye_distance = np.linalg.norm([right_eye_x - left_eye_x, right_eye_y - left_eye_y])

            # Tính góc xoay của mặt
            angle = np.degrees(np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))

            # Kích thước kính râm dựa trên khoảng cách giữa hai mắt
            glasses_width = int(2.0 * eye_distance)  # Điều chỉnh tỉ lệ
            glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

            # Resize kính râm
            resized_glasses = cv2.resize(sunglasses, (glasses_width, glasses_height))

            # Tọa độ đặt kính râm
            center_x = (left_eye_x + right_eye_x) // 2
            center_y = (left_eye_y + right_eye_y) // 2

            # Tạo ma trận xoay
            M = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), angle, 1.0)
            rotated_glasses = cv2.warpAffine(resized_glasses, M, (glasses_width, glasses_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            # Lật kính theo chiều dọc
            rotated_glasses = cv2.flip(rotated_glasses, 0)

            # Lấy vùng kính râm xoay
            x1 = max(0, center_x - glasses_width // 2)
            x2 = min(w, center_x + glasses_width // 2)
            y1 = max(0, center_y - glasses_height // 2)
            y2 = min(h, center_y + glasses_height // 2)

            # Điều chỉnh vùng overlay nếu kính bị cắt bớt
            overlay_width = x2 - x1
            overlay_height = y2 - y1
            rotated_glasses = cv2.resize(rotated_glasses, (overlay_width, overlay_height))

            # Overlay kính râm lên khung hình
            alpha_s = rotated_glasses[:, :, 3] / 255.0  # Kênh alpha của kính
            alpha_l = 1.0 - alpha_s

            for c in range(3):  # Duyệt qua các kênh màu (BGR)
                frame[y1:y2, x1:x2, c] = (
                    alpha_s * rotated_glasses[:, :, c] +
                    alpha_l * frame[y1:y2, x1:x2, c]
                )

    # Hiển thị video
    cv2.imshow('Face Filter with Flip', frame)

    # Thoát bằng phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
