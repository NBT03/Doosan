import time

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import SetModbusOutput

class ModbusOutputSetter(Node):

    def __init__(self):
        super().__init__('modbus_output_setter')
        # Sửa lại tên dịch vụ, loại bỏ khoảng trắng thừa
        self.cli = self.create_client(SetModbusOutput, '/dsr01/modbus/set_modbus_output')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def set_modbus_output(self, name, value):
        req = SetModbusOutput.Request()
        req.name = name
        req.value = value

        self.get_logger().info(f'Setting modbus output: {name}, Value: {value}')
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Successfully set modbus output: {name} = {value}')
        else:
            self.get_logger().error('Failed to set modbus output')


def main(args=None):
    rclpy.init(args=args)

    modbus_output_setter = ModbusOutputSetter()

    # Set giá trị output cho 3 chân Modbus
    modbus_output_setter.set_modbus_output('gripper_signal_1', 2304)  # Ví dụ: output_1 = 1
    modbus_output_setter.set_modbus_output('gripper_signal_3', 0)  # Ví dụ: output_3 = 1
    modbus_output_setter.set_modbus_output('gripper_signal_5', 25000)  # Ví dụ: output_2 = 0
    time.sleep(2)
    modbus_output_setter.set_modbus_output('gripper_signal_3', 50)  # Ví dụ: output_3 = 1
    modbus_output_setter.set_modbus_output('gripper_signal_5', 25000)  # Ví dụ: output_2 = 0
    rclpy.shutdown()


if __name__ == '__main__':
    main()
