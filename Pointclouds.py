import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d

# Khởi tạo PyBullet và thiết lập môi trường
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Thêm sàn và một số vật thể
plane_id = p.loadURDF("plane.urdf")
cube_start_pos = [0, 0, 0.5]
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("assets/tote/toteA_large.urdf", [0, 0, 0.5])
robot_id1 = p.loadURDF("assets/objects/cube.urdf", [0, 0.1, 0.6])
robot_id2 = p.loadURDF("assets/objects/custom.urdf", [0.1, 0, 0.6])
robot_id3 = p.loadURDF("assets/objects/rod.urdf", [0, 0.3, 0.6])

# Hàm thiết lập camera và chụp ảnh RGB và depth
def get_camera_images(width, height):
    fov = 60
    aspect = width / height
    near = 0.1
    far = 10.0

    # Đặt camera thẳng đứng từ trên cao xuống nhìn về (0, 0, 0)
    camera_eye_position = [0, 0, 3]  # Vị trí camera cao trên trục z
    camera_target_position = [0, 0, 0]  # Mục tiêu của camera tại gốc tọa độ
    camera_up_vector = [1, 0, 0]  # Hướng "lên" của camera để đảm bảo nhìn thẳng xuống

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye_position,
        cameraTargetPosition=camera_target_position,
        cameraUpVector=camera_up_vector
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far
    )

    # Chụp ảnh RGB và depth
    _, _, rgb_image, depth_buffer, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    # Chuyển đổi depth buffer sang depth map
    depth = np.array(depth_buffer).reshape((height, width))
    depth = far * near / (far - (far - near) * depth)

    # Chuyển đổi ảnh RGB sang định dạng numpy
    rgb_array = np.array(rgb_image).reshape((height, width, 4))
    rgb = rgb_array[:, :, :3]  # Bỏ kênh alpha

    return rgb, depth


# Chuyển đổi ảnh depth và RGB sang point cloud
def depth_and_rgb_to_pointcloud(depth, rgb, fx, fy, cx, cy):
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = depth
    points = np.stack((x * z, y * z, z), axis=-1)

    # Chuyển đổi RGB sang định dạng phù hợp cho Open3D
    rgb_flat = rgb.reshape(-1, 3) / 255.0  # Chuyển sang phạm vi [0, 1]
    points_flat = points.reshape(-1, 3)

    return points_flat, rgb_flat


# Hiển thị point cloud với Open3D
def display_pointcloud(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


# Lấy ảnh RGB và depth từ PyBullet
width, height = 640, 480
rgb, depth = get_camera_images(width, height)

# Thông số nội tại của camera
fx = fy = width / (2 * np.tan(np.radians(60) / 2))
cx = width / 2
cy = height / 2

# Chuyển đổi depth và RGB sang point cloud
points, colors = depth_and_rgb_to_pointcloud(depth, rgb, fx, fy, cx, cy)

# Hiển thị point cloud
display_pointcloud(points, colors)

# Đóng kết nối PyBullet
p.disconnect()
