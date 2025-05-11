#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # file paths
    nav2_params = '/home/matt/Desktop/luna/src/astro_launch/params/nav2_params.yaml'
    arena_map   = '/home/matt/Desktop/luna/src/astro_launch/params/arena_map.yaml'
    urdf_file   = '/home/matt/Desktop/luna/src/astro_launch/urdf/rover.urdf'

    # 1. robot_state_publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['cat ', urdf_file])
        }]
    )

    # 2. static_transform_publisher (map → odom)
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        output='screen',
        arguments=[
            '0', '0', '0',   # x y z
            '0', '0', '0',   # roll pitch yaw
            'map',           # parent
            'odom'           # child
        ]
    )

    # 3. motor_command_node
    motor_node = Node(
        package='astro_launch',
        executable='motor_command_node',
        name='motor_command_node',
        output='screen'
    )

    # 4. RealSense camera (include external launch)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare('realsense2_camera').find('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={'align_depth.enable': 'true'}.items()
    )

    # 5. filtering node
    filtering_node = Node(
        package='open_cv_pkg',
        executable='filtering',
        name='filtering',
        output='screen'
    )

    # 6. og node
    og_node = Node(
        package='open_cv_pkg',
        executable='og',
        name='og',
        output='screen'
    )

    # 7-12. Nav2 core nodes
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        output='screen',
        parameters=[nav2_params]
    )
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        output='screen',
        parameters=[nav2_params]
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[nav2_params]
    )



    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        output='screen',
        parameters=[nav2_params]
    )
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'yaml_filename': arena_map
        }]
    )
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params]
    )

    return LaunchDescription([
        robot_state_publisher,
        static_map_to_odom,
        motor_node,
        realsense_launch,
        filtering_node,
        og_node,
        bt_navigator,
        planner_server,
        controller_server,
        lifecycle_manager,
        map_server,
        behavior_server,
    ])
