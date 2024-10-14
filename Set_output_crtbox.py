#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import SetCtrlBoxDigitalOutput

class SetDigitalOutputClient(Node):

    def __init__(self):
        super().__init__('set_digital_output_client')
        self.client = self.create_client(SetCtrlBoxDigitalOutput, '/dsr01/io/set_ctrl_box_digital_output')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Waiting for service /dsr01/io/set_ctrl_box_digital_output...')

    def send_request(self, index, value):
        request = SetCtrlBoxDigitalOutput.Request()
        request.index = index
        request.value = value

        # Send the request and wait for the response
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"Set output index {index} to value {value} successfully.")
        else:
            self.get_logger().error(f"Failed to set output index {index} to value {value}.")

def main(args=None):
    rclpy.init(args=args)

    # Create the client node
    digital_output_client = SetDigitalOutputClient()

    # Set the desired index and value
    index = 7  # You can change this value
    value = 0
    # You can change this value

    # Call the service
    digital_output_client.send_request(index, value)

    # Shutdown the client node
    digital_output_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
