import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
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
    rviz_config = os.path.join(pkg_share, 'rviz', 'g1.rviz')

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_content}],
    )

    jsp_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_args,
    )

    return [rsp_node, jsp_node, rviz_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model',
            default_value='g1_29dof_rev_1_0.urdf',
            description='URDF filename in the g1_description share directory',
        ),
        OpaqueFunction(function=launch_nodes),
    ])
