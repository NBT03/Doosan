from __future__ import division
from os import link
import sim
import pybullet as p
import random
import numpy as np
import math

MAX_ITERS = 10000  # số lần lặp tối đa cho thuật toán RRT*
delta_q = 0.3  # kích thước bước cho thuật toán

class Node:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent

def visualize_path(q_1, q_2, env, color=[0, 0, 0]):  # hiển thị đường đi giữa 2 cấu hình khớp q_1, q_2
    env.set_joint_positions(q_1.joint_positions)
    point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    env.set_joint_positions(q_2.joint_positions)
    point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def rrt_star(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env, distance=0.155, radius=1.0):
    V, E = [Node(q_init)], []
    path, found = [], False
    q_goal_node = Node(q_goal)

    for i in range(MAX_ITERS):
        q_rand = semi_random_sample(steer_goal_p, q_goal)
        q_nearest = nearest(V, q_rand)
        q_new = steer(q_nearest.joint_positions, q_rand, delta_q)
        q_new_node = Node(q_new, q_nearest)

        if not env.check_collision(q_new, distance):
            near_nodes_list = near(V, q_new_node, radius)
            q_new_node = choose_parent(V, near_nodes_list, q_nearest, q_new_node)
            V.append(q_new_node)
            E.append((q_nearest, q_new_node))
            visualize_path(q_nearest, q_new_node, env)
            rewire(V, E, near_nodes_list, q_new_node, env, distance)

            if get_euclidean_distance(q_goal, q_new) < delta_q:
                V.append(q_goal_node)
                E.append((q_new_node, q_goal_node))
                visualize_path(q_new_node, q_goal_node, env)
                found = True
                break

    if found:
        current_q = q_goal_node
        path.append(current_q.joint_positions)
        while current_q != V[0]:  # nút gốc (q_init)
            for edge in E:
                if edge[1] == current_q:
                    current_q = edge[0]
                    path.append(current_q.joint_positions)
        path.reverse()
        return path
    else:
        return None

def semi_random_sample(steer_goal_p, q_goal):
    prob = random.random()
    if prob < steer_goal_p:
        return q_goal
    else:
        q_rand = [random.uniform(-np.pi, np.pi) for _ in range(len(q_goal))]
    return q_rand

def get_euclidean_distance(q1, q2):
    distance = sum((q2[i] - q1[i])**2 for i in range(len(q1)))
    return math.sqrt(distance)

def nearest(V, q_rand):
    distance = float("inf")
    q_nearest = None
    for v in V:
        dist = get_euclidean_distance(q_rand, v.joint_positions)
        if dist < distance:
            q_nearest = v
            distance = dist
    return q_nearest

def steer(q_nearest, q_rand, delta_q):
    if get_euclidean_distance(q_rand, q_nearest) <= delta_q:
        return q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / get_euclidean_distance(q_rand, q_nearest) for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
        return q_new

def near(V, q_new, radius):
    return [v for v in V if get_euclidean_distance(v.joint_positions, q_new.joint_positions) < radius]

def choose_parent(V, near_nodes, q_nearest, q_new):
    min_cost = get_cost(q_nearest) + get_euclidean_distance(q_nearest.joint_positions, q_new.joint_positions)
    best_parent = q_nearest
    for node in near_nodes:
        cost = get_cost(node) + get_euclidean_distance(node.joint_positions, q_new.joint_positions)
        if cost < min_cost:
            min_cost = cost
            best_parent = node
    q_new.parent = best_parent
    return q_new

def rewire(V, E, near_nodes, q_new, env, distance):
    for node in near_nodes:
        if node != q_new.parent:
            new_cost = get_cost(q_new) + get_euclidean_distance(q_new.joint_positions, node.joint_positions)
            if new_cost < get_cost(node) and not env.check_collision(node.joint_positions, distance):
                for edge in E:
                    if edge[1] == node:
                        E.remove(edge)
                        break
                E.append((q_new, node))
                node.parent = q_new

def get_cost(node):
    cost = 0
    while node.parent is not None:
        cost += get_euclidean_distance(node.joint_positions, node.parent.joint_positions)
        node = node.parent
    return cost

def get_grasp_position_angle(object_id):
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle

if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim.PyBulletSim(object_shapes=object_shapes)
    num_trials = 3

    # PART 1: Basic robot movement
    passed = 0
    for i in range(num_trials):
        random_position = env._workspace1_bounds[:, 0] + 0.15 + \
            np.random.random_sample((3)) * (env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
        random_orientation = np.random.random_sample((3)) * np.pi / 4 - np.pi / 8
        random_orientation[1] += np.pi
        random_orientation = p.getQuaternionFromEuler(random_orientation)
        marker = sim.SphereMarker(position=random_position, radius=0.03, orientation=random_orientation)
        env.move_tool(random_position, random_orientation)
        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
        link_marker = sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[0, 1, 0, 0.8])
        delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
        delta_orn = np.max(np.abs(np.array(link_state[1]) - random_orientation))
        if delta_pos <= 1e-3 and delta_orn <= 1e-3:
            passed += 1
        env.step_simulation(1000)
        env.robot_go_home()
        del marker, link_marker
    print(f"[Robot Movement] {passed} / {num_trials} cases passed")

    # PART 2: Grasping
    passed = 0
    env.load_gripper()
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        object_z = p.getBasePositionAndOrientation(object_id)[0][2]
        if object_z >= 0.2:
            passed += 1
        env.reset_objects()
    print(f"[Grasping] {passed} / {num_trials} cases passed")

    # PART 3: RRT* Implementation
    passed = 0
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            path_conf = rrt_star(env.robot_home_joint_config,
                                 env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print("No collision-free path is found within the time budget. Continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.01)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim.SphereMarker(link_state[0], radius=0.02))

                print("Path executed. Dropping the object")
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()
                for joint_state in reversed(path_conf):
                    env.move_joints(joint_state, speed=0.01)
                markers = None
            p.removeAllUserDebugItems()
        env.robot_go_home()

        # Test if the object was actually transferred to the second bin
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and \
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and \
            object_pos[2] <= 0.2:
                passed += 1
        env.reset_objects()
    print(f"[RRT* Execution] {passed} / {num_trials} cases passed")
