#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import GetModbusInput

class ModbusInputGetter(Node):
    def __init__(self):
        super().__init__('modbus_input_getter')
        self.cli = self.create_client(GetModbusInput, '/dsr01/modbus/get_modbus_input')

        # Chờ cho dịch vụ có sẵn
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service /dsr01/modbus/get_modbus_input...')

    def get_modbus_input(self, signal_names):
        for name in signal_names:
            # Tạo yêu cầu
            req = GetModbusInput.Request()
            req.name = name  # Đặt tên theo tín hiệu

            self.get_logger().info(f'Calling get_modbus_input service for {name}...')
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                response = future.result()
                self.get_logger().info(f'Response for {name}: {response}')
            else:
                self.get_logger().error(f'Failed to call service get_modbus_input for {name}')

def main(args=None):
    rclpy.init(args=args)

    modbus_input_getter = ModbusInputGetter()

    # Danh sách tín hiệu Modbus mà bạn muốn lấy giá trị
    signal_names = ['gripper_signal_1', 'gripper_signal_2', 'gripper_signal_3',
                    'gripper_signal_4', 'gripper_signal_5', 'gripper_signal_6']

    # Gọi dịch vụ để lấy input Modbus cho tất cả tín hiệu
    modbus_input_getter.get_modbus_input(signal_names)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
