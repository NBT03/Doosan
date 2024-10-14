#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import ConfigDeleteModbus

class ModbusDeleter(Node):
    def __init__(self):
        super().__init__('modbus_deleter')
        self.cli = self.create_client(ConfigDeleteModbus, '/dsr01/modbus/config_delete_modbus')

        # Chờ cho dịch vụ có sẵn
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service /dsr01/modbus/config_delete_modbus...')

    def delete_modbus(self, signal_names):
        for name in signal_names:
            # Tạo yêu cầu
            req = ConfigDeleteModbus.Request()
            req.name = name  # Đặt tên theo tín hiệu

            self.get_logger().info(f'Calling config_delete_modbus service for {name}...')
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                response = future.result()
                self.get_logger().info(f'Response for {name}: {response}')
            else:
                self.get_logger().error(f'Failed to call service config_delete_modbus for {name}')

def main(args=None):
    rclpy.init(args=args)

    modbus_deleter = ModbusDeleter()

    # Danh sách tín hiệu Modbus mà bạn muốn xóa
    signal_names = ['gripper_signal_1', 'gripper_signal_2', 'gripper_signal_3',
                    'gripper_signal_4', 'gripper_signal_5', 'gripper_signal_6']

    # Gọi dịch vụ để xóa input Modbus cho tất cả tín hiệu
    modbus_deleter.delete_modbus(signal_names)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
