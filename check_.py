import pybullet as p
import pybullet_data
import time

class RobotCollisionTester:
    def __init__(self, urdf_path):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.robot_body_id = p.loadURDF(urdf_path)
        self.obstacles = self.create_obstacles()

    def create_obstacles(self):
        # Tạo một số chướng ngại vật đơn giản
        obstacles = []
        obstacle_id = p.loadURDF("cube.urdf", [1, 0, 0.5], globalScaling=0.5)
        obstacles.append(obstacle_id)
        return obstacles

    def set_joint_positions(self, q):
        for i, value in enumerate(q):
            p.resetJointState(self.robot_body_id, i, value)

    def check_collision(self, q, distance=0.18):
        self.set_joint_positions(q)
        for obstacle_id in self.obstacles:
            closest_points = p.getClosestPoints(self.robot_body_id, obstacle_id, distance)
            if closest_points is not None and len(closest_points) != 0:
                return True
        return False

    def disconnect(self):
        p.disconnect()

# Đường dẫn đến tệp URDF của robot Doosan
urdf_path = "assets/ur5/doosan_origin.urdf"

# Tạo đối tượng kiểm tra va chạm
collision_tester = RobotCollisionTester(urdf_path)

# Kiểm tra va chạm với một số vị trí khớp khác nhau
test_joint_positions = [
    [0, 0, 0, 0, 0, 0],  # Vị trí khớp ban đầu
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Một vị trí khớp khác
    [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],  # Một vị trí khớp khác
]

for q in test_joint_positions:
    if collision_tester.check_collision(q):
        print(f"Collision detected for joint positions: {q}")
    else:
        print(f"No collision for joint positions: {q}")

# Ngắt kết nối PyBullet
