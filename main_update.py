from __future__ import division
import random
from traceback import print_tb

import numpy as np
import math
import pybullet as p
import sim_update
import threading
import time
from scipy.spatial import cKDTree

MAX_ITERS = 10000
delta_q = 0.1  # Step size

class Node:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent


def visualize_path(q_1, q_2, env, color=[0, 1, 0], tree='start'):
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    debug_color = color if tree == 'start' else [1, 0, 1]  # Màu sắc khác nhau cho hai cây
    p.addUserDebugLine(point_1, point_2, debug_color, 1.0)



def bidirectional_rrt(env, q_start, q_goal, MAX_ITERS, delta_q, steer_goal_p, max_connection_distance=0.15):
    """
    Thuật toán Bidirectional RRT với tích hợp APF: Xây dựng hai cây từ điểm bắt đầu và mục tiêu,
    sau đó cố gắng kết nối chúng khi có bất kỳ cặp nút nào gần nhau.

    :param env: Môi trường mô phỏng (đối tượng PyBulletSim).
    :param q_start: Cấu hình khớp nối ban đầu (6 chiều).
    :param q_goal: Cấu hình khớp nối mục tiêu (6 chiều).
    :param MAX_ITERS: Số lượng vòng lặp tối đa.
    :param delta_q: Kích thước bước đi.
    :param steer_goal_p: Xác suất để mẫu hóa hướng tới mục tiêu.
    :param max_connection_distance: Khoảng cách tối đa để kết nối hai cây.
    :return: Đường đi nếu tìm thấy, ngược lại là None.
    """
    # Khởi tạo hai cây
    tree_start = [Node(q_start)]
    tree_goal = [Node(q_goal)]


    for i in range(MAX_ITERS):
        # Luân phiên mở rộng giữa cây start và cây goal
        if i % 2 == 0:
            current_tree = tree_start
            other_tree = tree_goal
            tree_identifier = 'start'
            steer_target = q_goal  # Định hướng mẫu hóa về mục tiêu khi mở rộng cây start
        else:
            current_tree = tree_goal
            other_tree = tree_start
            tree_identifier = 'goal'
            steer_target = q_start  # Định hướng mẫu hóa về khởi đầu khi mở rộng cây goal


        # Mẫu hóa một cấu hình ngẫu nhiên với định hướng tương ứng và ảnh hưởng từ APF
        q_rand = semi_random_sample(
            steer_goal_p,
            steer_target,
        )

        # Tìm cấu hình gần nhất trong cây hiện tại
        q_nearest = nearest([node.joint_positions for node in current_tree], q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)

        # Kiểm tra đường đi không va chạm
        if not env.check_collision(q_new, distance=0.125):
            # Thêm nút mới vào cây hiện tại
            new_node = Node(q_new)
            nearest_node = next(node for node in current_tree if node.joint_positions == q_nearest)
            new_node.parent = nearest_node
            current_tree.append(new_node)
            visualize_path(q_nearest, q_new, env, tree=tree_identifier)

            # Thử kết nối hai cây với công thức mới
            combined_path = connect_trees(tree_start, tree_goal, env, max_connection_distance)
            if combined_path:
                print(f"Đã tìm thấy đường đi tại vòng lặp {i}")
                return combined_path

        else:
            print(f"Phát hiện va chạm với q_new tại vòng lặp {i}")

        if i % 500 == 0:
            print(f"Vòng lặp {i}: Đã mở rộng các cây.")

    print("Thuật toán Bidirectional RRT không tìm thấy đường đi trong giới hạn vòng lặp.")
    return None


def semi_random_sample(steer_goal_p, steer_target):

    prob = random.random()
    if prob < steer_goal_p:
        return steer_target
    else:
        q_rand = [random.uniform(-np.pi, np.pi) for i in range(len(steer_target))]
    return q_rand



def get_euclidean_distance(q1, q2):
    distance = 0
    for i in range(len(q1)):
        distance += (q2[i] - q1[i])**2
    return math.sqrt(distance)



def nearest(V, q_rand):
    distance = float("inf")
    q_nearest = None
    for v in V:
        if get_euclidean_distance(q_rand, v) < distance:
            q_nearest = v
            distance = get_euclidean_distance(q_rand, v)
    return q_nearest


def steer(q_nearest, q_rand, delta_q):
    q_new = None
    if get_euclidean_distance(q_rand, q_nearest) <= delta_q:
        q_new = q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / get_euclidean_distance(q_rand, q_nearest) for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
    return q_new
def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle
def run_bidirectional_rrt():
    """
    Run the Bidirectional RRT algorithm within the simulation.
    """
    env.load_gripper()
    passed = 0
    for _ in range(100):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            path_conf = bidirectional_rrt(env, env.robot_home_joint_config,
                                         env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5)
            print("Path configuration:", path_conf)
            if path_conf is None:
                print("No collision-free path is found within the iteration limit. Continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.01)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim_update.SphereMarker(link_state[0], radius=0.01))
                print("Path executed. Dropping the object")
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                path_conf1 = path_conf[::-1]
                if path_conf1:
                    for joint_state in path_conf1:
                        env.move_joints(joint_state, speed=0.01)
                        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                        markers.append(sim_update.SphereMarker(link_state[0], radius=0.01))
                markers = None
            p.removeAllUserDebugItems()

        env.robot_go_home()
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if (-0.8 <= object_pos[0] <= -0.2) and (-0.3 <= object_pos[1] <= 0.3) and (object_pos[2] <= 0.2):
            passed += 1
        env.reset_objects()

def draw():
    print("Starting draw function")
    # Initialize line IDs
    line_ids = [None, None, None]
    current_object_id = None
    current_obstacle_id = None
    
    def get_distant(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    while True:
        try:
            # Ensure there are objects and obstacles in the environment
            if len(env._objects_body_ids) == 0 or len(env.obstacles) == 0:
                print("No objects or obstacles found, waiting...")
                time.sleep(0.1)
                continue
            
            # Get current object and obstacle IDs
            object_id = env._objects_body_ids[0]
            obstacles_id = env.obstacles[0]
            
            # Check if the object or obstacle has been reset
            if object_id != current_object_id or obstacles_id != current_obstacle_id:
                # Reset line IDs when a new object or obstacle is detected
                line_ids = [None, None, None]
                current_object_id = object_id
                current_obstacle_id = obstacles_id
                print(f"Detected object or obstacles change. Updated object_id: {object_id}, obstacle_id: {obstacles_id}")
            
            # Verify that the object and obstacle still exist
            try:
                p.getBodyInfo(object_id)
                p.getBodyInfo(obstacles_id)
            except p.error as e:
                print(f"Body ID error: {e}")
                time.sleep(0.1)
                continue
            
            # Get link positions
            getlink1 = p.getLinkState(object_id, 0)[0]
            getlink2 = p.getLinkState(object_id, 1)[0]
            midpoint = np.add(getlink1, getlink2) / 2
            
            # Find the closest points between the object and obstacle
            closest_points = p.getClosestPoints(obstacles_id, object_id, 100)
            if not closest_points:
                print("No closest points found")
                a = getlink1  # Assign a default value to avoid errors
            else:
                a = closest_points[0][5]
                
            # Calculate the distance and print it
            distance = get_distant(midpoint, a)
            print(f"Distance between midpoint and closest point: {distance}")
            
            # Define lines to be drawn
            lines_to_draw = [
                (getlink1, a, [1, 0, 0]),    # Red line
                (getlink2, a, [1, 0, 0]),    # Green line
                (midpoint, a, [0, 1, 0])     # Green line
            ]
            
            # Draw or update debug lines
            for i, (start, end, color) in enumerate(lines_to_draw):
                if line_ids[i] is None:
                    # Create a new debug line if it doesn't exist
                    line_ids[i] = p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2)
                else:
                    # Update the existing debug line
                    p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2, replaceItemUniqueId=line_ids[i])
            
        except IndexError as e:
            print(f"IndexError: {e}. Possible object or obstacle indices out of range. Retrying...")
            # Reset line IDs to reinitialize lines in the next iteration
            line_ids = [None, None, None]
            current_object_id = None
            current_obstacle_id = None
        except Exception as e:
            print(f"Exception in draw: {e}")
        
        # Pause briefly to reduce CPU usage
        time.sleep(0.1)

def is_path_collision_free(env, q_start, q_end, step_size=0.01, distance=0.15):
    distance_between = get_euclidean_distance(q_start, q_end)
    steps = int(distance_between / step_size)
    for i in range(1, steps + 1):
        interpolated_q = [
            q_start[j] + (q_end[j] - q_start[j]) * (i / steps)
            for j in range(len(q_start))
        ]
        if env.check_collision(interpolated_q, distance=distance):
            return False
    return True

def connect_trees(tree_start, tree_goal, env, max_connection_distance):
    """
    Kết nối hai cây RRT nếu có bất kỳ cặp nút nào trong hai cây có khoảng cách nhỏ hơn ngưỡng cho phép 
    và đường đi giữa chúng không gặp va chạm, sử dụng KD-Tree để tối ưu hóa.

    :param tree_start: Danh sách các Node trong cây start.
    :param tree_goal: Danh sách các Node trong cây goal.
    :param env: Đối tượng môi trường PyBulletSim.
    :param max_connection_distance: Khoảng cách tối đa để kết nối hai nút.
    :return: Đường đi kết hợp nếu kết nối thành công, ngược lại None.
    """
    # Tạo KD-Tree cho tree_goal
    tree_goal_positions = [node.joint_positions for node in tree_goal]
    kd_tree_goal = cKDTree(tree_goal_positions)

    for node_start in tree_start:
        q_start = node_start.joint_positions
        # Tìm tất cả các nút trong tree_goal gần node_start hơn max_connection_distance
        indices = kd_tree_goal.query_ball_point(q_start, r=max_connection_distance)
        for idx in indices:
            node_goal = tree_goal[idx]
            distance = get_euclidean_distance(q_start, node_goal.joint_positions)
            print(f"Đã tìm thấy cặp nút gần nhau với khoảng cách {distance:.4f}. Thử kết nối...")
            
            # Kiểm tra va chạm giữa hai nút
            steps = int(distance / (max_connection_distance / 2))
            collision = False
            for t in range(1, steps + 1):
                interp_q = [
                    q_start[j] + 
                    (node_goal.joint_positions[j] - q_start[j]) * (t / steps)
                    for j in range(len(q_start))
                ]
                if env.check_collision(interp_q, distance=0.15):
                    print("Đường đi giữa hai nút gặp va chạm. Không kết nối.")
                    collision = True
                    break
            if not collision:
                # Xây dựng đường đi từ cây start
                path_start = []
                current = node_start
                while current is not None:
                    path_start.append(current.joint_positions)
                    current = current.parent
                path_start.reverse()

                # Xây dựng đường đi từ cây goal
                path_goal = []
                current = node_goal
                while current is not None:
                    path_goal.append(current.joint_positions)
                    current = current.parent

                # Kết hợp cả hai đường đi
                combined_path = path_start + path_goal
                print("Hai cây đã được kết nối thành công.")
                return combined_path

    print("Không tìm thấy cặp nút nào để kết nối hai cây.")
    return None

if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/rod.urdf",
    ]
    env = sim_update.PyBulletSim(object_shapes=object_shapes)
    thread1 = threading.Thread(target = run_bidirectional_rrt)
    thread2 = threading.Thread(target = draw)
    thread1.start()
    thread2.start()





















