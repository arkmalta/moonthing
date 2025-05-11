#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from std_msgs.msg import Bool


class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')

        # Subscriber to non-ground points (dynamic obstacles)
        self.sub = self.create_subscription(
            PointCloud2,
            '/non_ground_points',
            self.pointcloud_callback,
            10
        )

        # New: start disabled
        self.publishing_enabled = False
        # Subscribe to an "enable" topic
        self.enable_sub = self.create_subscription(
            Bool, '/grid_publish_enable', self.enable_callback, 10
        )

        # Publisher for the occupancy grid
        self.occ_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        # --- NEW: Publisher for the filtered-point cloud ---
        self.cloud_pub = self.create_publisher(PointCloud2, '/grid_cloud', 10)

        

        # Timer for rate limiting publication
        self.timer = self.create_timer(1.0, self.publish_occupancy_grid)  # Publish every 2 seconds

        # Storage for grid processing
        self.counter_grid = None  # Grid holding the observation count for each cell

        # Counter method parameters
        self.counter_threshold = 3  # Number of consecutive observations required to mark as occupied
        self.decay_value = 1        # How much to decrement the counter when a cell is not observed

        # Grid parameters
        self.grid_size = 100            # Dimensions: grid_size x grid_size cells
        self.resolution = 0.1           # Grid cell size (meters per cell)
        self.map_origin = [-2.5, -2.5]  # Map origin (X, Y) in meters
        self.frame_id = "camera_link"   # Frame in which the occupancy grid & cloud are published

        self.get_logger().info('Counter-Based Occupancy Grid Node started.')

        
    def enable_callback(self, msg: Bool):
        self.publishing_enabled = msg.data
        state = "enabled" if msg.data else "disabled"
        self.get_logger().info(f"Grid publishing {state} by external command")

    def pointcloud_callback(self, msg):
        # Extract x,y and build a raw occupancy mask
        points = np.array([
            (p[0], p[1]) for p in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        ], dtype=np.float32)

        raw_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        for x, y in points:
            gx = int((x - self.map_origin[0]) / self.resolution)
            gy = int((y - self.map_origin[1]) / self.resolution)
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                raw_grid[gy, gx] = 100

        binary_grid = np.where(raw_grid == 100, 100, 0).astype(np.int8)

        if self.counter_grid is None:
            self.counter_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        inc = (binary_grid == 100)
        dec = (binary_grid == 0)

        self.counter_grid[inc] += 1
        self.counter_grid[dec] = np.maximum(self.counter_grid[dec] - self.decay_value, 0)

    def publish_occupancy_grid(self):
        if not self.publishing_enabled or self.counter_grid is None:
            return

        # Build persistent grid
        persistent_grid = np.where(self.counter_grid >= self.counter_threshold, 100, 0).astype(np.int8)

        # --- Build & publish OccupancyGrid ---
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.frame_id

        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_size
        grid_msg.info.height = self.grid_size
        grid_msg.info.origin.position.x = self.map_origin[0]
        grid_msg.info.origin.position.y = self.map_origin[1]
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        grid_msg.data = persistent_grid.flatten().tolist()
        self.occ_pub.publish(grid_msg)
        self.get_logger().info("Published Counter-Based Occupancy Grid")

        # --- NEW: Build & publish PointCloud2 of occupied cells ---
        pts = []
        w = self.grid_size
        res = self.resolution
        ox, oy = self.map_origin

        for idx, val in enumerate(persistent_grid.flatten()):
            if val == 100:
                x = (idx % w) * res + ox + res/2.0
                y = (idx // w) * res + oy + res/2.0
                pts.append((x, y, 0.0))

        cloud_msg = pc2.create_cloud_xyz32(grid_msg.header, pts)
        self.cloud_pub.publish(cloud_msg)

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
