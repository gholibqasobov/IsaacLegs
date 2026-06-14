# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch.substitutions import (
    LaunchConfiguration,
    IfElseSubstitution,
    TextSubstitution,
)
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    policy_path = os.path.join(
        get_package_share_directory('fullbody_controller'),
        'policy', 'g1_locomotion', 'policy.pt'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "policy_path",
            default_value=policy_path,
            description="path to the exported TorchScript policy (IO_descriptors.yaml must sit beside it, "
                        "or set io_descriptors_path)"),
        DeclareLaunchArgument(
            "io_descriptors_path",
            default_value="",
            description="path to Isaac Lab's IO_descriptors.yaml "
                        "(default: <policy dir>/IO_descriptors.yaml)"),
        DeclareLaunchArgument(
            "decimation",
            default_value="1",
            description="run the policy every Nth tick; with 50 Hz sensors, 1 -> 50 Hz control"),
        DeclareLaunchArgument(
            "odom_twist_in_body_frame",
            default_value="True",
            description="True if /odom twist is in the body frame (REP-103); False to rotate from world"),
        DeclareLaunchArgument(
            "warmup_sec",
            default_value="0.0",
            description="seconds to ease the robot into the policy's default pose before engaging it "
                        "(0 disables)"),
        DeclareLaunchArgument(
            "warmup_interpolate",
            default_value="True",
            description="interpolate from the measured spawn pose to default over warmup_sec (no snap)"),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        DeclareLaunchArgument(
            "namespace",
            default_value="g1_01",
            description="ROS namespace for the G1 controller"),
        DeclareLaunchArgument(
            "use_namespace",
            default_value="False",
            description="Whether to apply the ROS namespace to the node"),
        Node(
            package='fullbody_controller',
            executable='policy_controller',
            name='fullbody_policy_controller',
            output="screen",
            namespace=IfElseSubstitution(
                [LaunchConfiguration('use_namespace')],
                [LaunchConfiguration('namespace')],
                [TextSubstitution(text='')]
            ),
            parameters=[{
                'policy_path': LaunchConfiguration('policy_path'),
                'io_descriptors_path': LaunchConfiguration('io_descriptors_path'),
                'decimation': ParameterValue(LaunchConfiguration('decimation'), value_type=int),
                'odom_twist_in_body_frame': ParameterValue(
                    LaunchConfiguration('odom_twist_in_body_frame'), value_type=bool),
                'warmup_sec': ParameterValue(LaunchConfiguration('warmup_sec'), value_type=float),
                'warmup_interpolate': ParameterValue(
                    LaunchConfiguration('warmup_interpolate'), value_type=bool),
                "use_sim_time": ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            }]

        ),
    ])
