#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import ConfigCreateModbus

class ModbusConfigurator(Node):
    def __init__(self):
        super().__init__('modbus_configurator')
        self.client = self.create_client(ConfigCreateModbus, '/dsr01/modbus/config_create_modbus')

        # Wait until the service is available
        while not self.client.wait_for_service(timeout_sec=1):
            self.get_logger().info('Waiting for service /dsr01/modbus/config_create_modbus...')

        # List of 6 configurations
        self.modbus_configs = [
            {'name': 'gripper_signal_1', 'ip': '192.168.137.11', 'port': 502, 'reg_type': 3, 'index': 0, 'value': 127, 'slave_id': 1},
            {'name': 'gripper_signal_2', 'ip': '192.168.137.11', 'port': 502, 'reg_type': 2, 'index': 0, 'value': 127, 'slave_id': 1},
            {'name': 'gripper_signal_3', 'ip': '192.168.137.11', 'port': 502, 'reg_type': 3, 'index': 1, 'value': 127, 'slave_id': 1},
            {'name': 'gripper_signal_4', 'ip': '192.168.137.11', 'port': 502, 'reg_type': 2, 'index': 1, 'value': 127, 'slave_id': 1},
            {'name': 'gripper_signal_5', 'ip': '192.168.137.11', 'port': 502, 'reg_type': 3, 'index': 2, 'value': 127, 'slave_id': 1},
            {'name': 'gripper_signal_6', 'ip': '192.168.137.11', 'port': 502, 'reg_type': 2, 'index': 2, 'value': 127, 'slave_id': 1},
        ]

        # Track the number of pending requests
        self.pending_requests = len(self.modbus_configs)

        # Call the service for each configuration
        for config in self.modbus_configs:
            self.send_request(config)

    def send_request(self, config):
        request = ConfigCreateModbus.Request()
        request.name = config['name']
        request.ip = config['ip']
        request.port = config['port']
        request.reg_type = config['reg_type']
        request.index = config['index']
        request.value = config['value']
        request.slave_id = config['slave_id']

        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Modbus configuration successful!')
            else:
                self.get_logger().error('Failed to configure Modbus!')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
        finally:
            # Decrement the number of pending requests
            self.pending_requests -= 1

            # If all requests are processed, stop the node
            if self.pending_requests == 0:
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    modbus_configurator = ModbusConfigurator()

    # Keep the node alive until configuration is done
    rclpy.spin(modbus_configurator)

    modbus_configurator.destroy_node()


if __name__ == '__main__':
    main()
