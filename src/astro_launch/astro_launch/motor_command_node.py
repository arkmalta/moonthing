#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

class MotorCommandNode(Node):
    def __init__(self):
        super().__init__('motor_command_node')

        # Subscribe to Nav2's output velocity commands
        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Publish motor command array: [m0,m1,m2,m3,m4,m5]
        self.motor_pub = self.create_publisher(
            Float32MultiArray, '/command_velocity', 10
        )

        self.get_logger().info("Motor Command Node Initialized")

        # ─── PATTERN CONFIGURATION ──────────────────────────────────────────
        # 1 tick of tank‐turn, then N ticks of normal curved mixing
        self.tank_ticks     = 1
        self.curve_ticks    = 1
        self.pattern_length = self.tank_ticks + self.curve_ticks
        self.pattern_index  = 0           # goes 0,1,...,pattern_length-1, then wraps

        # ─── STATIC TUNING PARAMS ────────────────────────────────────────────
        self.slip_factor       = 1.0     # multiplies raw commands to counter slip
        self.rover_width       = 0.5     # meters, axle‐to‐axle distance
        self.ang_thresh        = 0.13     # rad/s: below this → “straight” mode
        self.lin_small_thresh  = 0.05    # m/s : below this → “pure turn” mode
        self.toggle_freq       = self.pattern_length * 2    # Hz  : pattern tick rate



        # ─── TIMING STATE ────────────────────────────────────────────────────
        self.toggle_period    = 1.0 / self.toggle_freq
        self.last_toggle_time = self.get_clock().now()

        # ─── STATE FOR PURE‐TURN OSCILLATION ─────────────────────────────────
        # still use a boolean flip for pure‐turn jiggles
        self.curve_mode = True

    def cmd_vel_callback(self, msg: Twist):
        linear  = msg.linear.x
        angular = msg.angular.z

        # 1) Compute the raw differential‐drive mix
        left_speed  = linear - (angular * self.rover_width / 2.0)
        right_speed = linear + (angular * self.rover_width / 2.0)

        # 2) Advance our timers/pattern once per toggle_period
        now = self.get_clock().now()
        dt  = (now - self.last_toggle_time).nanoseconds * 1e-9
        if dt >= self.toggle_period:
            # flip pure‐turn boolean
            self.curve_mode = not self.curve_mode
            # step pattern index for moderate‐curve
            self.pattern_index = (self.pattern_index + 1) % self.pattern_length
            self.last_toggle_time = now

        # are we in the "tank-turn" sub-tick?
        in_tank_phase = (self.pattern_index < self.tank_ticks)

        # 3) Mode selection
        if abs(angular) < self.ang_thresh:
            # ─── Mode 1: Straight / gentle curve ───────────────────────────
            cmd_left, cmd_right = left_speed, right_speed

        elif abs(linear) >= self.lin_small_thresh:
            # ─── Mode 2: Moderate curve with pattern ──────────────────────
            if in_tank_phase:
                # tank‐turn branch (scaled down)
                mag    = max(abs(left_speed), abs(right_speed), 1e-6)
                sign_l = 1 if left_speed  >= 0 else -1
                sign_r = 1 if right_speed >= 0 else -1
                cmd_left  =  (mag * sign_l) / 1.5
                cmd_right = -(mag * sign_r) / 1.5
            else:
                # smooth curve branch
                cmd_left, cmd_right = left_speed, right_speed

        else:
            # ─── Mode 3: Pure turn ────────────────────────────────────────
            if self.curve_mode:
                # tiny forward nudge to break static friction
                forward_mag = self.lin_small_thresh
                cmd_left, cmd_right = forward_mag, forward_mag
            else:
                # real in-place spin
                turn_mag = abs(angular * self.rover_width / 2.0)
                turn_mag = max(turn_mag, self.lin_small_thresh)
                sign_a = 1 if angular >= 0 else -1
                cmd_left  =  turn_mag * sign_a
                cmd_right = -turn_mag * sign_a

        # 4) Apply slip compensation & normalize to [-1, 1]
        cmd_left  *= self.slip_factor
        cmd_right *= self.slip_factor
        norm = max(abs(cmd_left), abs(cmd_right), 1.0)
        left_motor  = cmd_left  / norm
        right_motor = cmd_right / norm

        # 5) Publish six‐wheel command (R/L/R/L/R/L)
        motor_msg = Float32MultiArray()
        motor_msg.data = [
            (1.0 * right_motor), (1.0 * left_motor),
            (1.0 * right_motor), (1.0 * left_motor),
            (1.1 * right_motor), (1.1 * left_motor)
        ]
        self.motor_pub.publish(motor_msg)

        self.get_logger().info(
            f"Powers L={left_motor:.2f} R={right_motor:.2f} "
            f"(pattern_idx={self.pattern_index}, pure_mode={'curve' if self.curve_mode else 'tank'})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MotorCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down motor_command_node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
