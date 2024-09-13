from __future__ import division
import random
import numpy as np
import math
import pybullet as p
# import sim
import sim_update
import threading
# Constants
MAX_ITERS = 10000
delta_q = 0.3  # Step size


class Node:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent


def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def dynamic_rrt_star(env, q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, velocities, distance=0.1):
    velocities = [[0.01, 0.01, 0.01],
                  [0.01, 0.05, -0.01],
                  [0.01, -0.01, 0.01],
                  [0.01, 0.01, 0],
                  [0.01, 0.01, 0],
                  [0.01, 0.01, 0],
                  [0.01, 0.01, 0]
                  ]
    V, E = [Node(q_init)], []
    path, found = [], False

    for i in range(MAX_ITERS):
        q_rand = semi_random_sample(steer_goal_p, q_goal)
        q_nearest = nearest([node.joint_positions for node in V], q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)

        if not env.check_collision(q_new, distance=0.15):
            q_new_node = Node(q_new)
            q_nearest_node = next(node for node in V if node.joint_positions == q_nearest)
            q_new_node.parent = q_nearest_node


            if q_new_node not in V:
                V.append(q_new_node)
            if (q_nearest_node, q_new_node) not in E:
                E.append((q_nearest_node, q_new_node))
                visualize_path(q_nearest, q_new, env)
            if get_euclidean_distance(q_goal, q_new) < delta_q:
                q_goal_node = Node(q_goal, q_new_node)
                V.append(q_goal_node)
                E.append((q_new_node, q_goal_node))
                visualize_path(q_new, q_goal, env)
                found = True
                break

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


def semi_random_sample(steer_goal_p, q_goal):
    prob = random.random()
    if prob < steer_goal_p:
        return q_goal
    else:
        q_rand = [random.uniform(-np.pi, np.pi) for _ in range(len(q_goal))]
    return q_rand


def get_euclidean_distance(q1, q2):
    return math.sqrt(sum((q2[i] - q1[i]) ** 2 for i in range(len(q1))))


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
    num_trials = 3
    passed = 0
    # for i in range(num_trials):
    #     random_position = env._workspace1_bounds[:, 0] + 0.15 + \
    #                       np.random.random_sample((3)) * (
    #                               env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
    #     random_orientation = np.random.random_sample((3)) * np.pi / 4 - np.pi / 8
    #     random_orientation[1] += np.pi
    #     random_orientation = p.getQuaternionFromEuler(random_orientation)
    #     marker = sim.SphereMarker(position=random_position, radius=0.03, orientation=random_orientation)
    #     env.move_tool(random_position, random_orientation)
    #     link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
    #     link_marker = sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1],
    #                                    rgba_color=[0, 1, 0, 0.8])
    #     delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
    #     delta_orn = np.max(np.abs(np.array(link_state[1]) - random_orientation))
    #     if delta_pos <= 1e-3 and delta_orn <= 1e-3:
    #         passed += 1
    #     env.step_simulation(1000)
    #     env.robot_go_home()
    #     del marker, link_marker
    # print(f"[Robot Movement] {passed} / {num_trials} cases passed")
    #
    # passed = 0
    env.load_gripper()
    # for _ in range(1):
    #     object_id = env._objects_body_ids[0]
    #     position, grasp_angle = get_grasp_position_angle(object_id)
    #     grasp_success = env.execute_grasp(position, grasp_angle)
    #     object_z = p.getBasePositionAndOrientation(object_id)[0][2]
    #     if object_z >= 0.2:
    #         passed += 1
    #     env.reset_objects()
    # print(f"[Grasping] {passed} / {num_trials} cases passed")

    passed = 0
    for _ in range(10):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            velocities = [0.01]  # Example velocity for moving obstacles
            path_conf = dynamic_rrt_star(env, env.robot_home_joint_config,
                                         env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, velocities)

            if path_conf is None:
                print("No collision-free path is found within the time budget. Continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.005)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))
                print("Path executed. Dropping the object")
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                path_conf1 = dynamic_rrt_star(env, env.robot_goal_joint_config,
                                              env.robot_home_joint_config, MAX_ITERS, delta_q, 0.5, velocities)
                if path_conf1:
                    for joint_state in path_conf1:
                        env.move_joints(joint_state, speed=0.005)
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
if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim_update.PyBulletSim(object_shapes=object_shapes)
    def move_ostacles():
       env.update_moving_obstacles()
    drrt = threading.Thread(target=run_dynamic_rrt_star)
    a = threading.Thread(target=move_ostacles)
    drrt.start()
    a.start()
    drrt.join()
    a.join()



