from __future__ import division
import random
import numpy as np
import math
import pybullet as p
import sim_update
import threading
import time
import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint

MAX_ITERS = 10000
delta_q = 0.1  # Step size
k_att = 0.55   # Coefficient for attractive force
k_rep = 1   # Coefficient for repulsive force
d0 = 0.1      # Distance threshold for repulsive force


class NodeROS(Node):
    def __init__(self):
        super().__init__('robot_trajectory_controller')
        self.client = self.create_client(MoveJoint, '/dsr01/motion/move_joint')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = MoveJoint.Request()

    def send_trajectory(self, trajectory):
        for joint_angles in trajectory:
            joint_angles_in_degrees = [math.degrees(angle) for angle in joint_angles]  # Chuyển đổi radian sang độ
            self.req.pos = joint_angles_in_degrees
            self.req.vel = 100.0  # Tốc độ di chuyển
            self.req.acc = 100.0  # Gia tốc
            self.req.time = 0.0  # Thời gian thực hiện lệnh
            self.req.radius = 0.0  # Bán kính chuyển động tròn nếu cần
            self.req.mode = 0  # Chế độ điều khiển
            self.req.blend_type = 1
            self.req.sync_type = 0

            future = self.client.call_async(self.req)
            rclpy.spin_until_future_complete(self, future)

            if future.result().success:
                self.get_logger().info('Successfully moved to: %s' % joint_angles_in_degrees)
            else:
                self.get_logger().error('Failed to move to: %s' % joint_angles_in_degrees)
            time.sleep(1.0)


class NodeSim:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent


def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def dynamic_rrt_star(env, q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, distance=0.1):
    V, E = [NodeSim(q_init)], []
    path, found = [], False

    for i in range(MAX_ITERS):
        q_rand = semi_random_sample(steer_goal_p, q_goal)
        q_nearest = nearest([node.joint_positions for node in V], q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)

        attractive = attractive_force(q_nearest, q_goal, k_att)
        obstacles = [node.joint_positions for node in V if node.joint_positions != q_nearest]
        repulsive = repulsive_force(q_nearest, obstacles, k_rep, d0)

        max_force = 0.75  # Giới hạn lực tổng (bạn có thể điều chỉnh giá trị này)
        total_force = apply_forces(q_nearest, q_goal, obstacles, k_att, k_rep, d0, max_force)

        # Apply forces to new point
        q_new = [q_new[i] + total_force[i] for i in range(len(q_new))]

        q_new = [q_new[i] + attractive[i] + repulsive[i] for i in range(len(q_new))]

        if not env.check_collision(q_new, distance=0.1):
            q_new_node = NodeSim(q_new)
            q_nearest_node = next(node for node in V if node.joint_positions == q_nearest)
            q_new_node.parent = q_nearest_node

            if q_new_node not in V:
                V.append(q_new_node)
            if (q_nearest_node, q_new_node) not in E:
                E.append((q_nearest_node, q_new_node))
                visualize_path(q_nearest, q_new, env)
            if get_euclidean_distance(q_goal, q_new) < delta_q:
                q_goal_node = NodeSim(q_goal, q_new_node)
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


def attractive_force(q_current, q_goal, k_att):
    return [k_att * (q_goal[i] - q_current[i]) for i in range(len(q_current))]


def repulsive_force(q_current, obstacles, k_rep, d0):
    force = [0.0] * len(q_current)
    for obs in obstacles:
        dist = get_euclidean_distance(q_current, obs)
        if dist < d0:
            repulsive_magnitude = k_rep * (1 / dist - 1 / d0) / (dist ** 2)
            for i in range(len(force)):
                force[i] += repulsive_magnitude * (q_current[i] - obs[i]) / dist
    return force
def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle
def apply_forces(q_nearest, q_goal, obstacles, k_att, k_rep, d0, max_force):
    attractive = attractive_force(q_nearest, q_goal, k_att)
    repulsive = repulsive_force(q_nearest, obstacles, k_rep, d0)

    # Tính tổng lực
    total_force = [attractive[i] + repulsive[i] for i in range(len(attractive))]

    # Giới hạn lực tổng theo max_force
    force_magnitude = np.linalg.norm(total_force)
    if force_magnitude > max_force:
        scaling_factor = max_force / force_magnitude
        total_force = [f * scaling_factor for f in total_force]

    return total_force

def run_dynamic_rrt_star():
    rclpy.init(args=None)
    ros_node = NodeROS()

    env.load_gripper()
    passed = 0
    for _ in range(10):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)

        if grasp_success:
            path_conf = dynamic_rrt_star(env, env.robot_home_joint_config,
                                         env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5)
            if path_conf:
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.05)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))
                ros_node.send_trajectory(path_conf)  # Gửi path_conf qua ROS2

                print("Path executed. Dropping the object")
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                path_conf1 = path_conf[::-1]
                if path_conf1:
                    for joint_state in path_conf1:
                        env.move_joints(joint_state, speed=0.05)
                        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                        markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))
                    ros_node.send_trajectory(path_conf1)  # Gửi path_conf1 qua ROS2

                markers = None
            p.removeAllUserDebugItems()
        env.robot_go_home()
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and \
                object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and \
                object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()
    rclpy.shutdown()


if __name__ == "__main__":
    random.seed(1)
    object_shapes = ["assets/objects/cube.urdf"]
    env = sim_update.PyBulletSim(object_shapes=object_shapes)
    run_dynamic_rrt_star()
