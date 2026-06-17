# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the policy controller for the Unitree G1 humanoid.

Thin wrapper around ``policy_controller.launch.py`` that points ``policy_path``
at the G1 checkpoint shipped under ``policy/g1_locomotion/`` in this package.
Pass through any other launch args by appending them on the command line,
e.g. ``ros2 launch fullbody_controller g1.launch.py decimation:=2``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory('fullbody_controller')
    base = os.path.join(share, 'launch', 'policy_controller.launch.py')
    policy_path = os.path.join(share, 'policy', 'g1_locomotion', 'policy.pt')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base),
            launch_arguments={'policy_path': policy_path}.items(),
        ),
    ])
