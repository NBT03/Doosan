import pybullet as p
import pybullet_data
import numpy as np
import cv2

# Khởi tạo môi trường PyBullet
p.connect(p.GUI)  # hoặc p.DIRECT nếu bạn không cần giao diện đồ họa
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Đường dẫn tới các tệp dữ liệu của PyBullet

# Tải mặt phẳng cơ bản
p.loadURDF("plane.urdf")

# Đặt trọng lực
p.setGravity(0, 0, -9.81)

# Tải một robot hoặc đối tượng nào đó vào mô phỏng
robot_id = p.loadURDF("assets/tote/toteA_large.urdf", [0, 0, 0.5])
robot_id1 = p.loadURDF("assets/objects/cube.urdf", [0, 0.1, 0.6])
robot_id2 = p.loadURDF("assets/objects/custom.urdf", [0.1, 0, 0.6])
robot_id3 = p.loadURDF("assets/objects/rod.urdf", [0, 0.3, 0.6])



# Thiết lập thông số camera
camera_target_position = [0, 0, 0.5]  # Vị trí mà camera sẽ nhìn vào
camera_distance = 1.5  # Khoảng cách từ camera đến tâm cảnh
camera_yaw = 50  # Góc quay của camera quanh trục Z
camera_pitch = -35  # Góc quay của camera quanh trục X
camera_roll = 0  # Góc quay của camera quanh trục Y
fov = 60  # Trường nhìn của camera (Field of View)
aspect = 1.0  # Tỷ lệ khung hình (width/height)
near_plane = 0.1  # Mặt phẳng gần
far_plane = 100.0  # Mặt phẳng xa

# Tính toán ma trận xem và ma trận chiếu cho camera
view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_target_position,
                                                  distance=camera_distance,
                                                  yaw=camera_yaw,
                                                  pitch=camera_pitch,
                                                  roll=camera_roll,
                                                  upAxisIndex=2)
projection_matrix = p.computeProjectionMatrixFOV(fov=fov,
                                                 aspect=aspect,
                                                 nearVal=near_plane,
                                                 farVal=far_plane)

# Lấy ảnh từ camera
width, height, rgb_image, depth_image, seg_image = p.getCameraImage(width=640,
                                                                   height=480,
                                                                   viewMatrix=view_matrix,
                                                                   projectionMatrix=projection_matrix)

# Chuyển đổi hình ảnh RGB thành định dạng có thể hiển thị bằng OpenCV
rgb_image = np.reshape(rgb_image, (height, width, 4))  # RGB + Alpha channel
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)

# Hiển thị ảnh sử dụng OpenCV
cv2.imshow("Camera View", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ngắt kết nối PyBullet
p.disconnect()
