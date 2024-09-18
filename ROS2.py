import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint
import time


class RobotTrajectoryController(Node):
    def __init__(self):
        super().__init__('robot_trajectory_controller')
        self.client = self.create_client(MoveJoint, '/dsr01/motion/move_joint')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = MoveJoint.Request()

    def send_trajectory(self, trajectory):
        for joint_angles in trajectory:
            self.req.pos = joint_angles
            self.req.vel = 2000.0  # Tốc độ di chuyển
            self.req.acc = 2000.0  # Gia tốc
            self.req.time = 0.0  # Thời gian thực hiện lệnh
            self.req.radius = 0.0  # Bán kính chuyển động tròn nếu cần
            self.req.mode = 0  # Chế độ điều khiển
            self.req.blend_type = 1
            self.req.sync_type = 0

            future = self.client.call_async(self.req)
            rclpy.spin_until_future_complete(self, future)

            if future.result().success:
                self.get_logger().info('Successfully moved to: %s' % joint_angles)
            else:
                self.get_logger().error('Failed to move to: %s' % joint_angles)

            # Chờ giữa các lần di chuyển nếu cần
            time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotTrajectoryController()

    # Danh sách các tập hợp góc khớp cần di chuyển
    trajectory_list = [
        [-94.06, -13.65, 101.87, 1.48, 89.39, 3.73],
        [-35.00, 25.65, 60.87, 1.48, 89.39, 3.73],

        # Các tập hợp góc khớp khác
    ]

    robot_controller.send_trajectory(trajectory_list)
    rclpy.shutdown()


if __name__ == '__main__':
    main()


