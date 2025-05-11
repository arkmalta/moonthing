import rclpy
from rclpy.node import Node
from opcua import Client, ua
from std_msgs.msg import Int32, String, Float32MultiArray
import traceback

class OpcUaClientNode(Node):
    def __init__(self):
        super().__init__('opcua_client')
       
        # Set up OPC UA connection
        self.client = Client("opc.tcp://192.168.50.3:49580")


        #Try security things to see if they work
        #self.client.set_security_string("Basic256Sha256,SignAndEncrypt")  # Match server policy
        #self.client.application_uri = "urn:example:client"  # Unique identifier for your client

        # Load server certificate (if required)
        #self.client.load_server_certificate("myrio_cert.der")  # Server's certificate file

        # Load client certificate (if mutual authentication is required)
        #self.client.load_client_certificate("client_cert.pem")
        #self.client.load_private_key("client_private_key.pem")


        try:
            self.client.connect()
            self.get_logger().info("Connected to OPC UA Server!")
        #except ConnectionRefusedError:
         #   self.get_logger().error("Server refused connection")
        #except ua.uaerrors.BadNoMatch:
         #   self.get_logger().error("Security policy mismatch")
        except Exception as e:
            self.get_logger().error(f"Connection failed: {e}")
            # Check if specific error message can be found
            self.get_logger().error(f"Connection failed: {str(e)}\n{traceback.format_exc()}")
       
        # Cache OPC UA node references for reading (Rio-owned variables)
        self.node_exca   = self.client.get_node("ns=2;s=OPCVariables.O_R_EXCAStatus")
        self.node_depo   = self.client.get_node("ns=2;s=OPCVariables.O_R_DEPOStatus")
        self.node_navi   = self.client.get_node("ns=2;s=OPCVariables.O_R_NAVIStatus")
        #self.node_auto   = self.client.get_node("ns=2;s=OPCVariables.Automation_State")
        self.node_current = self.client.get_node("ns=2;s=OPCVariables.O_R_Current")
        self.node_encoder = self.client.get_node("ns=2;s=OPCVariables.O_R_Encoder")
       
        # Cache OPC UA node references for writing (Pi-owned variables)
        self.node_pi_command    = self.client.get_node("ns=2;s=OPCVariables.O_P_PiCommand")
        self.node_command_vel   = self.client.get_node("ns=2;s=OPCVariables.O_P_CommandVel")
        self.node_complete_task = self.client.get_node("ns=2;s=OPCVariables.O_P_CompleteTask")
        #self.node_servo_up = self.client.get_node("ns=2;s=OPCVariables.O_P_ServoUp")

        # -------------------------
        # Publishers (reading from OPC UA, publishing to ROS)
        # -------------------------
        self.exca_pub    = self.create_publisher(Int32, 'excavation_status', 10)
        self.depo_pub    = self.create_publisher(Int32, 'deposition_status', 10)
        self.navi_pub    = self.create_publisher(Int32, 'navigation_status', 10)
        #self.auto_pub    = self.create_publisher(String, 'automation_state', 10)
        self.current_pub = self.create_publisher(Float32MultiArray, 'motor_current', 10)
        self.encoder_pub = self.create_publisher(Float32MultiArray, 'encoder_values', 10)

        # -------------------------
        # Subscribers (receiving from ROS, writing to OPC UA)
        # -------------------------
        self.pi_command_sub = self.create_subscription(
            Int32,
            'pi_command',
            self.write_pi_command,
            10
        )

        # self.servo_up_sub = self.create_subscription(
        #     Int32,
        #     'servo_up',
        #     self.write_servo_up,
        #     10
        # )
        self.command_vel_sub = self.create_subscription(
            Float32MultiArray,
            'command_velocity',
            self.write_command_vel,
            10
        )
        self.complete_task_sub = self.create_subscription(
            Int32,
            'complete_task',
            self.write_complete_task,
            10
        )

        # Timer to read all OPC UA variables every second
        self.timer = self.create_timer(0.1, self.read_all_opcua_data)

    def read_all_opcua_data(self):
        """Reads multiple OPC UA variables and publishes them to ROS topics.

        OPC UA variables read:
          - OPCVariables.O_R_EXCAStatus: Excavation status
          - OPCVariables.O_R_DEPOStatus: Deposition status
          - OPCVariables.O_R_NAVIStatus: Navigation status
          - OPCVariables.Automation_State: Automation state (string)
          - OPCVariables.O_R_Current: Current values for the 6 wheel motors
          - OPCVariables.O_R_Encoder: Encoder values for the 6 wheel motors
        """
        try:
            # Read Excavation status
            exca_value = self.node_exca.get_value()
            msg_exca = Int32()
            msg_exca.data = int(exca_value)
            self.exca_pub.publish(msg_exca)
            self.get_logger().info(f"OPC UA Excavation status: {exca_value}")

            # Read Deposition status
            depo_value = self.node_depo.get_value()
            msg_depo = Int32()
            msg_depo.data = int(depo_value)
            self.depo_pub.publish(msg_depo)
            self.get_logger().info(f"OPC UA Deposition status: {depo_value}")

            # Read Navigation status
            navi_value = self.node_navi.get_value()
            msg_navi = Int32()
            msg_navi.data = int(navi_value)
            self.navi_pub.publish(msg_navi)
            self.get_logger().info(f"OPC UA Navigation status: {navi_value}")

            # Read Automation State (string)
            #auto_value = self.node_auto.get_value()
            #msg_auto = String()
            #msg_auto.data = str(auto_value)
            #self.auto_pub.publish(msg_auto)
            #self.get_logger().info(f"OPC UA Automation state: {auto_value}")

            # Read Motor Current values and explicitly convert each to float
            current_value = self.node_current.get_value()
            msg_current = Float32MultiArray()
            msg_current.data = [float(val) for val in current_value]
            self.current_pub.publish(msg_current)
            self.get_logger().info(f"OPC UA Motor current: {current_value}")

            # Read Encoder values and explicitly convert each to float
            encoder_value = self.node_encoder.get_value()
            msg_encoder = Float32MultiArray()
            msg_encoder.data = [float(val) for val in encoder_value]
            self.encoder_pub.publish(msg_encoder)
            self.get_logger().info(f"OPC UA Encoder values: {encoder_value}")

        except Exception as e:
            self.get_logger().error(f"Failed to read OPC UA data: {e}")

    def write_pi_command(self, msg):
        """
        Writes the Pi command to the OPC UA variable OPCVariables.O_P_PiCommand.
       
        Value Definitions:
          10 - NAVI
          20 - EXCA
          30 - DEPO
          99 - Error, return to Waiting on Command
        """
        try:
            variant_value = ua.Variant(msg.data, ua.VariantType.Int32)
            self.node_pi_command.set_value(variant_value)
            self.get_logger().info(f"Sent Pi Command to OPC UA: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Failed to write Pi Command: {e}")

    def write_command_vel(self, msg):
        """
        Writes the command velocity for 6 motors to the OPC UA variable OPCVariables.O_P_CommandVel.
       
        The message contains 6 values (one per motor) with values between -1 and 1.
        """
        try:
            float_values = [float(x) for x in msg.data]
            variant_value = ua.Variant(float_values, ua.VariantType.Double)
            self.node_command_vel.set_value(variant_value)
            self.get_logger().info(f"Sent Command Velocity to OPC UA: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Failed to write Command Velocity: {e}")

    def write_complete_task(self, msg):
        """
        Writes the complete task signal to the OPC UA variable OPCVariables.O_P_CompleteTask.
       
        Value Definitions:
          1 - EXCAStatus, DEPOStatus, and NAVIStatus remain the same
          0 - Reset EXCAStatus, DEPOStatus, and NAVIStatus to 0
        """
        try:
            variant_value = ua.Variant(msg.data, ua.VariantType.Int32)
            self.node_complete_task.set_value(variant_value)
            self.get_logger().info(f"Sent Complete Task to OPC UA: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Failed to write Complete Task: {e}")

    # def write_servo_up(self, msg: Int32):
    #     """
    #     Writes the servo-up command to OPCVariables.O_P_ServoUp.
    #     Expect an Int32 on /servo_up indicating how far to move the servo.
    #     """
    #     try:
    #         variant = ua.Variant(msg.data, ua.VariantType.Int32)
    #         self.node_servo_up.set_value(variant)
    #         self.get_logger().info(f"Sent ServoUp to OPC UA: {msg.data}")
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to write ServoUp: {e}")


    def destroy_node(self):
        """Cleanly disconnect from the OPC UA server upon node shutdown."""
        try:
            self.client.disconnect()
            self.get_logger().info("Disconnected from OPC UA Server")
        except Exception as e:
            self.get_logger().error(f"Error disconnecting from OPC UA Server: {e}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OpcUaClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()