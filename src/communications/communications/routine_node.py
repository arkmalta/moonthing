#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Int32, Bool
from nav2_msgs.action import NavigateToPose
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Quaternion

# OPC-UA Status Codes
STATUS_NOT_STARTED = 0
STATUS_ACK         = 10
STATUS_STARTED     = 20
STATUS_FINISHED    = 30
STATUS_ERROR       = 99

# PiCommand codes
CMD_NAVI = 10
CMD_EXCA = 20
CMD_DEPO = 30

# Sub-state machine
SUBSTATES = {
    'IDLE':         0,
    'WAIT_IDLE':    1,
    'SEND_CMD':     2,
    'WAIT_ACK':     3,
    'WAIT_STARTED': 4,
    'EXECUTING':    5,
    'COMPLETING':   6,
    'EYES_ON':   7,
    'EYES_WAIT': 8,
}

class Task:
    NAV        = 'nav'     
    NAV_EYES   = 'nav_eyes'  
    EXCA = 'exca'
    DEPO = 'depo'

    def __init__(self, ttype, waypoints=None):
        self.type      = ttype
        self.waypoints = waypoints or []
        self.wp_index  = 0


class RoutineNode(Node):
    def __init__(self):
        super().__init__('routine_node')
        self.grid_pub = self.create_publisher(Bool, '/grid_publish_enable', 10)

        # Task queue + tracking
        self.tasks            = []
        self.current_task     = None
        self.substate         = SUBSTATES['IDLE']
        self.task_retries     = 0

        # OPC-UA statuses
        self.navigation_status = STATUS_NOT_STARTED
        self.excavation_status = STATUS_NOT_STARTED
        self.deposition_status = STATUS_NOT_STARTED

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 server unavailable!")
            raise RuntimeError("Navigation server connection failed")

        # Publishers & subscribers
        self.pi_cmd_pub   = self.create_publisher(Int32, 'pi_command',    10)
        self.complete_pub = self.create_publisher(Int32, 'complete_task', 10)

        self.create_subscription(Int32, 'navigation_status', self._navi_cb, 10)
        self.create_subscription(Int32, 'excavation_status', self._exca_cb, 10)
        self.create_subscription(Int32, 'deposition_status', self._depo_cb, 10)

        # Run at 2 Hz
        self.create_timer(0.5, self._run_state_machine)

    # ─── Public API ───────────────────────────────────────────────
    def add_navigation(self, waypoints):
        self.tasks.append(Task(Task.NAV, waypoints))
    def add_excavation(self):
        self.tasks.append(Task(Task.EXCA))
    def add_deposition(self):
        self.tasks.append(Task(Task.DEPO))

    # ─── Core loop ────────────────────────────────────────────────
    def _run_state_machine(self):
        # 1) pick next task if idle
        if self.substate == SUBSTATES['IDLE'] and not self.current_task and self.tasks:
            self.current_task = self.tasks.pop(0)
            self.substate     = SUBSTATES['WAIT_IDLE']
            self.task_retries = 0
            self.current_task.wp_index = 0
            self.get_logger().info(f"→ Starting task '{self.current_task.type}'")

        # 2) if a task is active, dispatch
        if not self.current_task:
            return

        if self.current_task.type in (Task.NAV, Task.NAV_EYES):
            self._handle_navigation()
        elif self.current_task.type == Task.EXCA:
            self._handle_excavation()
        elif self.current_task.type == Task.DEPO:
            self._handle_deposition()
        else:
            self._abort_task()

    # ─── Navigation sub-state machine ────────────────────────────
    def _handle_navigation(self):
        st     = self.substate
        status = self.navigation_status

        if st == SUBSTATES['WAIT_IDLE']:
            self._send_complete(0)     
            if status == STATUS_NOT_STARTED:
                self.substate = SUBSTATES['SEND_CMD']

        elif st == SUBSTATES['SEND_CMD']:
            self._send_complete(0)
            self._send_pi(CMD_NAVI)
            self.substate = SUBSTATES['WAIT_ACK']

        elif st == SUBSTATES['WAIT_ACK']:
            if status == STATUS_ACK:
                self._send_complete(1)
                self._send_pi(0)
                self.substate = SUBSTATES['WAIT_STARTED']
            elif status == STATUS_ERROR:
                self._handle_failure()

        elif st == SUBSTATES['WAIT_STARTED']:
            if status == STATUS_STARTED:
                # With-eyes branch?
                if self.current_task.type == Task.NAV_EYES:
                    self.grid_pub.publish(Bool(data=True))     # enable scan
                    self.eyes_off_time = self.get_clock().now() + Duration(seconds=5)
                    self.substate = SUBSTATES['EYES_WAIT']
                else:
                    self._send_nav_goal()
                    self.substate = SUBSTATES['EXECUTING']

        elif st == SUBSTATES['EYES_WAIT']:
            if self.get_clock().now() >= self.eyes_off_time:
                self.grid_pub.publish(Bool(data=False))        # disable scan
                self._send_nav_goal()
                self.substate = SUBSTATES['EXECUTING']


        elif st == SUBSTATES['EXECUTING']:
            # wait for action callback
            pass

        elif st == SUBSTATES['COMPLETING']:
            if status == STATUS_FINISHED:          # 30
                self._send_complete(0)             # tell Rio we saw it
            elif status == STATUS_NOT_STARTED:     # 0  (Rio has cleared)
                self.current_task = None
                self.substate     = SUBSTATES['IDLE']


    # ─── Excavation (same pattern) ───────────────────────────────
    def _handle_excavation(self):
        st     = self.substate
        status = self.excavation_status

        if st == SUBSTATES['WAIT_IDLE']:
            self._send_complete(0)
            if status == STATUS_NOT_STARTED:
                self.substate = SUBSTATES['SEND_CMD']

        elif st == SUBSTATES['SEND_CMD']:
            self._send_complete(0)
            self._send_pi(CMD_EXCA)
            self.substate = SUBSTATES['WAIT_ACK']

        elif st == SUBSTATES['WAIT_ACK']:
            if status == STATUS_ACK:
                self._send_complete(1)
                self._send_pi(0)
                self.substate = SUBSTATES['EXECUTING']
            elif status == STATUS_ERROR:
                self._handle_failure()

        elif st == SUBSTATES['EXECUTING']:
            if status == STATUS_FINISHED:
                self.substate = SUBSTATES['COMPLETING']

        elif st == SUBSTATES['COMPLETING']:
            if status == STATUS_FINISHED:          # 30
                self._send_complete(0)             # tell Rio we saw it
            elif status == STATUS_NOT_STARTED:     # 0  (Rio has cleared)
                self.current_task = None
                self.substate     = SUBSTATES['IDLE']


    # ─── Deposition (same pattern) ───────────────────────────────
    def _handle_deposition(self):
        st     = self.substate
        status = self.deposition_status

        if st == SUBSTATES['WAIT_IDLE']:
            self._send_complete(0)
            if status == STATUS_NOT_STARTED:
                self.substate = SUBSTATES['SEND_CMD']

        elif st == SUBSTATES['SEND_CMD']:
            self._send_complete(0)
            self._send_pi(CMD_DEPO)
            self.substate = SUBSTATES['WAIT_ACK']

        elif st == SUBSTATES['WAIT_ACK']:
            if status == STATUS_ACK:
                self._send_complete(1)
                self._send_pi(0)
                self.substate = SUBSTATES['EXECUTING']
            elif status == STATUS_ERROR:
                self._handle_failure()

        elif st == SUBSTATES['EXECUTING']:
            if status == STATUS_FINISHED:
                self.substate = SUBSTATES['COMPLETING']

        elif st == SUBSTATES['COMPLETING']:
            if status == STATUS_FINISHED:          # 30
                self._send_complete(0)             # tell Rio we saw it
            elif status == STATUS_NOT_STARTED:     # 0  (Rio has cleared)
                self.current_task = None
                self.substate     = SUBSTATES['IDLE']

    # ─── Nav2 plumbing ──────────────────────────────────────────
    def _send_nav_goal(self):
        t = self.current_task
        x,y,th = t.waypoints[t.wp_index]
        goal = NavigateToPose.Goal()
        pst  = PoseStamped()
        pst.header.frame_id = 'map'
        pst.header.stamp    = self.get_clock().now().to_msg()
        pst.pose.position.x = x
        pst.pose.position.y = y
        q = Quaternion()
        q.z = math.sin(th/2.0)
        q.w = math.cos(th/2.0)
        pst.pose.orientation = q
        goal.pose = pst

        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self._nav_response)

    def _nav_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Nav goal rejected")
            self._handle_failure()
            return
        handle.get_result_async().add_done_callback(self._nav_result)

    def _nav_result(self, future):
        status = future.result().status
        if status == 4:  # SUCCEEDED
            t = self.current_task
            if t.wp_index < len(t.waypoints)-1:
                t.wp_index += 1
                self._send_nav_goal()
            else:
                self._send_complete(49)
                self.substate = SUBSTATES['COMPLETING']
        else:
            self.get_logger().error(f"Nav2 failed (status {status})")
            self._handle_failure()

    # ─── Error / retry / abort ───────────────────────────────────
    def _handle_failure(self):
        if self.task_retries < 3:
            self.task_retries += 1
            self.get_logger().warn(f"Retry {self.task_retries}/3")
            self.substate = SUBSTATES['WAIT_IDLE']
        else:
            self.get_logger().error("Too many failures, aborting task")
            self._abort_task()

    def _abort_task(self):
        self._send_pi(0)
        self._send_complete(0)
        self.current_task = None
        self.substate     = SUBSTATES['IDLE']
        self.task_retries = 0

    # ─── ROS callbacks ───────────────────────────────────────────
    def _navi_cb(self, msg): self.navigation_status = msg.data
    def _exca_cb(self, msg): self.excavation_status = msg.data
    def _depo_cb(self, msg): self.deposition_status = msg.data

    # ─── Helpers ─────────────────────────────────────────────────
    def _send_pi(self, v):
        self.pi_cmd_pub.publish(Int32(data=v))
    def _send_complete(self, v):
        self.complete_pub.publish(Int32(data=v))
    def add_navigation_with_eyes(self, waypoints):
        self.tasks.append(Task(Task.NAV_EYES, waypoints))



def main(args=None):
    rclpy.init(args=args)
    node = RoutineNode()

    # build your routine:
    node.add_navigation([(1.35,2.2,0.0), (1.0,2.0,0.0)])
    node.add_excavation()
    node.add_navigation([(5.0,1.0,1.57)])
    node.add_deposition()
    node.add_navigation([(1.35, 2.2, 0.0)])
    node.add_navigation_with_eyes([(0.5, 1.2, 1.5)])


    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()