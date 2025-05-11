#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Quaternion, PoseStamped
import serial
import time
import datetime
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from collections import deque


class RoverLocalization(Node):
    def __init__(self):
        super().__init__('rover_localization')

        # EWMA smoothing factor: how much weight to give the newest measurement
        self.declare_parameter('ewma_alpha', 0.15)         # you can tweak this at runtime
        self.alpha = self.get_parameter('ewma_alpha').value

        # EWMA state: None until the first measurement arrives
        self.ewma_position = None    # will hold (x,y,z)
        self.ewma_heading  = None    # in radians


        self.tf_broadcaster = TransformBroadcaster(self)


        # Serial port parameters.
        self.baudrate = 115200
        # Define two serial port names. These ports are not assumed to correspond to a specific tag.
        self.port1 = "/dev/ttyACM0"
        self.port2 = "/dev/ttyACM1"

        # Initialize serial connections for both ports.
        self.serial_conn_1 = self.initialize_dwm(self.port1)
        self.serial_conn_2 = self.initialize_dwm(self.port2)

        # Dictionary to hold the latest positions from tags.
        # Keys will be tag IDs (e.g., "B0", "B1") and values will be tuples (x, y, z).
        self.tag_positions = {}

        # Publishers for the transformation matrix, quaternion, and pose stamped messages.
        self.matrix_publisher = self.create_publisher(Float64MultiArray, '/localization_matrix', 10)
        self.quaternion_publisher = self.create_publisher(Quaternion, '/localization_quaternion', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/rover_pose', 10)

        # Timer to read serial data and publish localization at 10 Hz.
        self.timer = self.create_timer(0.1, self.read_and_publish_pose)

    def initialize_dwm(self, port):
        """i
        Initialize a serial connection on the given port.
        Opens the serial port, sends carriage returns to ensure we are in shell mode, etc.
        """
        # try:
        #     dwm = serial.Serial(port=port, baudrate=self.baudrate, timeout=1)
        #     self.get_logger().info(f"Connected to {dwm.name}")
        #     # Send carriage returns to get into shell mode.
        #     dwm.write("\r\r".encode())
        #     time.sleep(1)
        #     return dwm
        # except serial.SerialException as e:
        #     self.get_logger().error(f"Failed to open serial port {port}: {e}")
        #     return None
        backoff = 1.0
        while rclpy.ok():
            try:
                dwm = serial.Serial(port=port,
                                    baudrate=self.baudrate,
                                    timeout=1)
                self.get_logger().info(f"Serial OK on {port}")
                dwm.write("\r\r".encode())
                time.sleep(1)
                return dwm
            except serial.SerialException as e:
                self.get_logger().warn(
                    f"Cannot open {port} ({e}), retrying in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff*2, 30.0)  # cap exponential back-off
        return None   # rclpy is shutting down

    def parse_position_data(self, line):
        """
        Parse a line of serial data from a tag.
        Expected format example:
          B0 T:34381140 POS:[220,2213,191,0] DIST0:...
        Returns:
          A tuple (tag_id, x, y, z) if parsing is successful, or None if parsing fails.
        """
        try:
            # Decode the byte string to a normal string and strip whitespace.
            decoded_line = line.decode().strip()
            # Split the line into tokens by spaces.
            tokens = decoded_line.split()
            if len(tokens) < 3:
                return None

            # The first token should be the tag ID (e.g., "B0" or "B1").
            tag_id = tokens[0]

            # Find the token that starts with "POS:[" containing the position data.
            pos_token = None
            for token in tokens:
                if token.startswith("POS:["):
                    pos_token = token
                    break
            if pos_token is None:
                return None

            # Extract the substring inside the square brackets.
            start = pos_token.find('[')
            end = pos_token.find(']')
            if start == -1 or end == -1:
                return None

            numbers_str = pos_token[start+1:end]
            # Split the numbers by commas.
            numbers = numbers_str.split(',')
            if len(numbers) < 3:
                return None

            # Convert the first three values to floats and convert from mm to m.
            x = float(numbers[0]) / 1000.0
            y = float(numbers[1]) / 1000.0
            z = float(numbers[2]) / 1000.0
            return tag_id, x, y, z
        except Exception as e:
            self.get_logger().warn(f"Error parsing position data: {e}")
            return None

    def read_serial(self, serial_conn, port):
        """
        Read one line from the given serial connection.
        Parses the line and returns the data (tag_id, x, y, z) if successful.
        Returns None if reading or parsing fails.
        """
        # if serial_conn:
        #     try:
        #         line = serial_conn.readline()
        #         if line:
        #             return self.parse_position_data(line)
        #     except Exception as e:
        #         self.get_logger().error(f"Error reading tag data: {e}")
        # return None
        if not serial_conn:
            return None

        try:
            line = serial_conn.readline()
        except serial.SerialException as e:
            self.get_logger().warn(f"Serial lost on {port}: {e}")
            serial_conn.close()
            # try to reconnect immediately
            return None

        if not line:
            return None

        parsed = self.parse_position_data(line)
        if parsed is None:
            self.get_logger().debug(f"Bad packet from {port}: {line!r}")
        return parsed

    def read_and_publish_pose(self):
        """
        Reads data from both serial connections, updates tag positions based on the tag ID,
        computes the midpoint and adjusted heading if valid data from both tags is available,
        and publishes the localization information on three topics:
          - /localization_matrix
          - /localization_quaternion
          - /rover_pose (PoseStamped)
        """
        # Read from the first serial connection.
        result1 = self.read_serial(self.serial_conn_1, self.port1)
        if result1 is not None:
            tag_id, x, y, z = result1
            # Update the tag position using the tag ID from the message.
            self.tag_positions[tag_id] = (x, y, z)
            self.get_logger().debug(f"Received {tag_id} from {self.port1}: x={x}, y={y}, z={z}")

        # Read from the second serial connection.
        result2 = self.read_serial(self.serial_conn_2, self.port2)
        if result2 is not None:
            tag_id, x, y, z = result2
            # Update the tag position using the tag ID from the message.
            self.tag_positions[tag_id] = (x, y, z)
            self.get_logger().debug(f"Received {tag_id} from {self.port2}: x={x}, y={y}, z={z}")

        # Check if positions from both required tags ("B0" for back and "B1" for front) are available.
        if "B0" in self.tag_positions and "B1" in self.tag_positions:
            back_pos = self.tag_positions["B0"]
            front_pos = self.tag_positions["B1"]

            # Check for NaN values in either tag's position.
            if (math.isnan(back_pos[0]) or math.isnan(back_pos[1]) or math.isnan(back_pos[2]) or
                math.isnan(front_pos[0]) or math.isnan(front_pos[1]) or math.isnan(front_pos[2])):
                self.get_logger().info("Position not shown: Received NaN values.")
                return  # Skip updating topics if data is invalid.

            # Compute the midpoint between the two tag positions.
            mid_x = (back_pos[0] + front_pos[0]) / 2.0
            mid_y = (back_pos[1] + front_pos[1]) / 2.0
            mid_z = (back_pos[2] + front_pos[2]) / 2.0

            # Compute the vector from the back tag (B0) to the front tag (B1).
            dx = front_pos[0] - back_pos[0]
            dy = front_pos[1] - back_pos[1]
            original_heading = math.atan2(dy, dx)
            # Adjust the heading 45° to the right (clockwise rotation by 45°).
            adjusted_heading = original_heading - (math.pi / 4)

            # === initialize EWMA on first sample ===
            if self.ewma_position is None:
                self.ewma_position = (mid_x, mid_y, mid_z)
                self.ewma_heading  = adjusted_heading
            else:
                # EWMA on position
                ex, ey, ez = self.ewma_position
                ex = self.alpha*mid_x + (1-self.alpha)*ex
                ey = self.alpha*mid_y + (1-self.alpha)*ey
                ez = self.alpha*mid_z + (1-self.alpha)*ez
                self.ewma_position = (ex, ey, ez)

                # EWMA on heading via sin/cos to avoid wrap issues
                sh = math.sin(self.ewma_heading)
                ch = math.cos(self.ewma_heading)
                s_new = math.sin(adjusted_heading)
                c_new = math.cos(adjusted_heading)
                s = self.alpha*s_new + (1-self.alpha)*sh
                c = self.alpha*c_new + (1-self.alpha)*ch
                self.ewma_heading = math.atan2(s, c)

            # rename for the rest of your code:
            avg_x, avg_y, avg_z = self.ewma_position
            avg_heading        = self.ewma_heading

			

            # log moving-average values
            self.get_logger().info(
                f"EWMA Midpoint: x={avg_x:.2f}, y={avg_y:.2f}, z={avg_z:.2f}; " #z should be avg_z
                f"EWMA Heading: {avg_heading:.2f} rad"
            )

            # Build a 4x4 homogeneous transformation matrix representing the pose.
            cos_a = math.cos(avg_heading)
            sin_a = math.sin(avg_heading)
            matrix_data = [
                cos_a, -sin_a, 0, avg_x,
                sin_a,  cos_a, 0, avg_y,
                0,      0,     1, avg_z,
                0,      0,     0, 1
            ] 
            # Ensure each value is a float.
            matrix_data = [float(value) for value in matrix_data]
            matrix_msg = Float64MultiArray()
            matrix_msg.data = matrix_data
            self.matrix_publisher.publish(matrix_msg)

            # Create a quaternion from the adjusted heading (assuming roll and pitch are 0).
            qz = math.sin(avg_heading / 2.0)
            qw = math.cos(avg_heading / 2.0)
            quat_msg = Quaternion()
            quat_msg.x = 0.0
            quat_msg.y = 0.0
            quat_msg.z = float(qz)
            quat_msg.w = float(qw)
            self.quaternion_publisher.publish(quat_msg)

            # Create a PoseStamped message containing the position and orientation.
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"  # Adjust as needed for your coordinate frame.
            # Set position using the computed midpoint.
            pose_msg.pose.position.x = avg_x
            pose_msg.pose.position.y = avg_y
            pose_msg.pose.position.z = avg_z
            # Set orientation using the computed quaternion (roll and pitch assumed to be 0).
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = float(qz)
            pose_msg.pose.orientation.w = float(qw)
            self.pose_publisher.publish(pose_msg)
            # Publish TF transform from odom → base_link
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "odom" #"map"
            t.child_frame_id = "tag_link" # "base_link"
            t.transform.translation.x = avg_x
            t.transform.translation.y = avg_y
            t.transform.translation.z = avg_z  # assuming flat surface

            # Use same quaternion as published Pose
            t.transform.rotation = pose_msg.pose.orientation

            # Send the transform
            self.tf_broadcaster.sendTransform(t)

        else:
            # If both tag positions have not yet been received, log a debug message.
            self.get_logger().debug("Waiting for both tag positions (B0 and B1).")

    def shutdown(self):
        """
        Clean up and close serial connections on shutdown.
        """
        if self.serial_conn_1:
            self.serial_conn_1.write("\r".encode())
            self.serial_conn_1.close()
            self.get_logger().info(f"Serial connection on {self.port1} closed.")
        if self.serial_conn_2:
            self.serial_conn_2.write("\r".encode())
            self.serial_conn_2.close()
            self.get_logger().info(f"Serial connection on {self.port2} closed.")

def main(args=None):
    rclpy.init(args=args)
    node = RoverLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()