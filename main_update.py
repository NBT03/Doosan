from __future__ import division
import random
import numpy as np
import math
import pybullet as p
import sim_update
import threading
import time

MAX_ITERS = 10000
delta_q = 0.1  # Step size

class Node:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent


def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    # env.set_joint_positions(q_1)
    # point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    # env.set_joint_positions(q_2)
    # point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    # p.addUserDebugLine(point_1, point_2, color, 1.0)
    pass


def dynamic_rrt_star(env, q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p):
    V, E = [Node(q_init)], []
    path, found = [], False

    for i in range(MAX_ITERS):
        # Lấy điểm gần nhất từ cây
        q_curr = V[-1].joint_positions

        # Lấy danh sách các vật cản trong môi trường
        obstacles = env.obstacles  # Giả sử env có hàm get_obstacles()

        # Sinh điểm ngẫu nhiên q_rand bằng APF
        q_rand = semi_random_sample(steer_goal_p, q_goal, q_curr, obstacles)

        # Tìm điểm gần nhất trong cây hiện tại
        q_nearest = nearest([node.joint_positions for node in V], q_rand)

        # Điều chỉnh q_rand về hướng q_nearest với khoảng cách delta_q
        q_new = steer(q_nearest, q_rand, delta_q)

        # Giới hạn bước di chuyển sau khi cộng thêm lực
        if get_euclidean_distance(q_new, q_nearest) > delta_q:
            q_new = steer(q_nearest, q_new, delta_q)

        print(q_new)
        # Kiểm tra va chạm
        if not env.check_collision(q_new,distance=0):
            q_new_node = Node(q_new)
            q_nearest_node = next(node for node in V if node.joint_positions == q_nearest)
            q_new_node.parent = q_nearest_node

            # Thêm vào cây và vẽ đường đi
            if q_new_node not in V:
                V.append(q_new_node)
            if (q_nearest_node, q_new_node) not in E:
                E.append((q_nearest_node, q_new_node))
                visualize_path(q_nearest, q_new, env)

            # Kiểm tra xem đã đạt đến đích chưa
            if get_euclidean_distance(q_goal, q_new) < delta_q:
                q_goal_node = Node(q_goal, q_new_node)
                V.append(q_goal_node)
                E.append((q_new_node, q_goal_node))
                visualize_path(q_new, q_goal, env)
                found = True
                break

    # Nếu tìm thấy đường đi, trả về đường dẫn
    if found:
        current_node = q_goal_node
        path.append(current_node.joint_positions)
        while current_node.parent is not None:
            current_node = current_node.parent
            path.append(current_node.joint_positions)
        path.reverse()
        return path
    else:
        return None


# Hàm semi_random_sample có tích hợp APF
def semi_random_sample(steer_goal_p, q_goal, q_curr, obstacles, eta_attractive=1.0, eta_repulsive=1, d0=0.5):
    prob = random.random()
    # Kiểm tra định dạng của q_curr và obs


    # Nếu xác suất nhỏ hơn steer_goal_p, chọn trực tiếp q_goal
    if prob < steer_goal_p:
        return q_goal
    else:
        # Lực hấp dẫn (từ đích)
        attractive_force = [eta_attractive * (q_goal[i] - q_curr[i]) for i in range(len(q_goal))]
        print(f"att {attractive_force}")
        def get_distant(a, b):
            return (math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))
        # Lực đẩy (từ vật cản)
        repulsive_force = [0.0] * len(q_goal)
        object = env._objects_body_ids[0]
        getlink1 = p.getLinkState(object, 0)[0]
        getlink2 = p.getLinkState(object, 1)[0]
        midpoint = np.add(getlink1, getlink2) / 2
        # print(midpoint)
        for obs in obstacles:
            # print(obs)
            distance_to_obstacle = p.getClosestPoints(obs,object,100)[0][8]
            print(f"dis {distance_to_obstacle}")
            if distance_to_obstacle < d0:  # Chỉ tính lực đẩy khi gần vật cản
                for i in range(len(q_goal)):
                        repulsive_force[i] += eta_repulsive * (1.0 / distance_to_obstacle - 1.0 / d0) * (1) / (distance_to_obstacle ** 2)
        print()
        # Tổng hợp lực (cộng lực hút và lực đẩy)
        total_force = [attractive_force[i] - repulsive_force[i] for i in range(len(q_goal))]

        # Tạo điểm q_rand mới chịu ảnh hưởng bởi lực tổng hợp
        q_rand = [q_curr[i] + total_force[i] for i in range(len(q_goal))]

        return q_rand



def get_euclidean_distance(q1, q2):
    # Kiểm tra nếu q1 và q2 đều là danh sách/mảng
    if isinstance(q1, list) and isinstance(q2, list) and len(q1) == len(q2):
        return math.sqrt(sum((q2[i] - q1[i]) ** 2 for i in range(len(q1))))
    else:
        raise ValueError("q1 và q2 phải là danh sách có cùng độ dài.")



def nearest(V, q_rand):
    distance = float("inf")
    q_nearest = None
    for v in V:
        if get_euclidean_distance(q_rand, v) < distance:
            q_nearest = v
            distance = get_euclidean_distance(q_rand, v)
    return q_nearest


def steer(q_nearest, q_rand, delta_q):
    if get_euclidean_distance(q_rand, q_nearest) <= delta_q:
        return q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / get_euclidean_distance(q_rand, q_nearest) for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
    return q_new
def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle
def run_dynamic_rrt_star():
    env.load_gripper()
    passed = 0
    for _ in range(100):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            path_conf = dynamic_rrt_star(env, env.robot_home_joint_config,
                                         env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5)
            print(path_conf)
            if path_conf is None:
                print("No collision-free path is found within the time budget. Continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.01)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))
                print("Path executed. Dropping the object")
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                path_conf1 = path_conf[::-1]
                if path_conf1:
                    for joint_state in path_conf1:
                        env.move_joints(joint_state, speed=0.01)
                        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                        markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))
                markers = None
            p.removeAllUserDebugItems()

        env.robot_go_home()
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and \
                object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and \
                object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

def draw():
    print("a")
    object = env._objects_body_ids[0]
    obstacles = env.obstacles[0]
    getlink1 = p.getLinkState(object, 0)[0]
    getlink2 = p.getLinkState(object, 1)[0]
    midpoint = np.add(getlink1, getlink2) / 2
    pointA = p.getClosestPoints(obstacles,object,100)
    a = pointA[0][5]
    line_id1 = p.addUserDebugLine(getlink1, a, lineColorRGB=[1, 0, 0], lineWidth=2)
    line_id2 = p.addUserDebugLine(getlink2, a, lineColorRGB=[0, 1, 0], lineWidth=2)
    line_id3 = p.addUserDebugLine(midpoint, a, lineColorRGB=[0, 1, 0], lineWidth=2)
    def get_distant(a,b):
        return (math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2))
    while True:
        getlink1 = p.getLinkState(object, 0)[0]
        getlink2 = p.getLinkState(object, 1)[0]
        midpoint = np.add(getlink1, getlink2) / 2
        pointA = p.getClosestPoints(obstacles, object, 100)
        a = pointA[0][5]
        print(get_distant(midpoint,a))
        p.addUserDebugLine(getlink1, a, lineColorRGB=[1, 0, 0], lineWidth=2,replaceItemUniqueId = line_id1)
        p.addUserDebugLine(getlink2, a, lineColorRGB=[0, 1, 0], lineWidth=2, replaceItemUniqueId = line_id2)
        p.addUserDebugLine(midpoint, a, lineColorRGB=[0, 1, 0], lineWidth=2,replaceItemUniqueId = line_id3)

if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/rod.urdf",
    ]
    env = sim_update.PyBulletSim(object_shapes=object_shapes)
    thread1 = threading.Thread(target = run_dynamic_rrt_star)
    thread2 = threading.Thread(target = draw)
    thread1.start()
    # thread2.start()