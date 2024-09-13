import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d

# Khởi tạo PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Tải một số đối tượng vào môi trường
planeId = p.loadURDF("plane.urdf")
objectId = p.loadURDF("duck_vhacd.urdf", [0, 0, 0.5])

# Chờ mô phỏng ổn định
for i in range(100):
    p.stepSimulation()
# Lấy thông tin về lưới bề mặt của vật thể
mesh_data = p.getMeshData(objectId, -1, flags=p.MESH_DATA_SIMULATION_MESH)

# Chuyển đổi các đỉnh lưới bề mặt thành point cloud
vertices = np.array(mesh_data[1])

# Tạo Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

# Hiển thị point cloud
o3d.visualization.draw_geometries([pcd])
p.disconnect()
