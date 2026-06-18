import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch_ros.actions import Node


def load_urdf(urdf_filename: str) -> str:
    pkg_share = get_package_share_directory('g1_description')
    urdf_path = os.path.join(pkg_share, 'urdf', urdf_filename)
    with open(urdf_path, 'r') as f:
        content = f.read()
    content = content.replace('filename="meshes/', 'filename="package://g1_description/meshes/')
    return content


def launch_nodes(context, *args, **kwargs):
    model_file = context.launch_configurations.get('model', 'g1_29dof_rev_1_0.urdf')
    urdf_content = load_urdf(model_file)

    pkg_share = get_package_share_directory('g1_description')
    rviz_config_override = context.launch_configurations.get('rviz_config', '')
    rviz_config = rviz_config_override if rviz_config_override else os.path.join(pkg_share, 'rviz', 'g1.rviz')

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_content}],
    )

    joint_state_pub_node = Node(
        package='g1_bringup',
        executable='g1_joint_state_pub',
        name='g1_joint_state_publisher',
        output='screen',
    )

    # The Livox MID-360 hardware clock runs ~20 s behind the host system clock.
    # Re-stamp every cloud with now() so TF lookups in pointcloud_to_laserscan
    # and rviz never see a timestamp that predates the TF buffer.
    livox_restamper_node = Node(
        package='g1_bringup_py',
        executable='livox_restamper',
        name='livox_restamper',
        output='screen',
    )

    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        parameters=[{
            # Height slice is in target_frame coordinates.
            # base_link == pelvis (~0.7 m above ground on G1), so
            # [-0.3, 0.3] captures a band from ~0.4 m to ~1.0 m above ground —
            # enough to see walls and most obstacles while ignoring the floor.
            'target_frame': 'base_link',
            'transform_tolerance': 0.2,
            'min_height': -0.3,
            'max_height': 0.3,
            'angle_min': -3.14159265,
            'angle_max':  3.14159265,
            'angle_increment': 0.00436,   # ~0.25 deg resolution
            'scan_time': 0.1,             # matches 10 Hz Livox publish rate
            'range_min': 0.5,             # ignore returns closer than 50 cm (robot body/arms)
            'range_max': 20.0,
            'use_inf': True,
            # Publish /scan with RELIABLE so nav2 (AMCL, costmaps) can subscribe to it.
            # pointcloud_to_laserscan defaults to BEST_EFFORT which is incompatible.
            'qos_overrides./scan.publisher.reliability': 'reliable',
            'qos_overrides./scan.publisher.durability': 'volatile',
        }],
        remappings=[
            ('cloud_in', '/utlidar/cloud_restamped'),
            ('scan',     '/scan'),
        ],
    )

    rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_args,
    )

    return [rsp_node, joint_state_pub_node, livox_restamper_node, pointcloud_to_laserscan_node, rviz_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model',
            default_value='g1_29dof_rev_1_0.urdf',
            description='URDF filename in the g1_description share directory',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='',
            description='Absolute path to a custom RViz config. Defaults to g1.rviz.',
        ),
        OpaqueFunction(function=launch_nodes),
    ])
