#!/usr/bin/env python3

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

"""ROS 2 controller that runs the trained policy in Isaac Sim.

* control rate = 50 Hz.

The exact joint order, default positions and action scaling are loaded from Isaac Lab's
``IO_descriptors.yaml`` (produced when the env is trained/exported with
``export_io_descriptors=True``), 
"""

import io
import os
import time

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState


class FullbodyController(Node):
    """Run policy: subscribe to robot state, publish joint position targets."""

    def __init__(self):
        """Initialize the FullbodyController node."""
        super().__init__('fullbody_policy_controller')

        # ---- parameters -----------------------------------------------------------------------------
        # default: resolved against the installed package share dir, so the node works from any cwd.
        # override with -p policy_path:=/abs/path/to/policy.pt to use a different checkpoint.
        default_policy_path = os.path.join(
            get_package_share_directory('fullbody_controller'),
            'policy', 'g1_locomotion', 'policy.pt')
        self.declare_parameter('policy_path', default_policy_path)
        # Isaac Lab IO descriptors yaml (joint order / defaults / scaling / obs layout);
        # default: ``IO_descriptors.yaml`` alongside the policy.
        self.declare_parameter('io_descriptors_path', '')
        # run the policy every Nth synchronized tick. With 50 Hz sensors, decimation=1 -> 50 Hz control.
        self.declare_parameter('decimation', 1)
        # warmup: before running the policy, drive the robot to the policy's default joint pose so the
        # first observation is in-distribution. seconds to hold/ease into default (0 disables warmup).
        self.declare_parameter('warmup_sec', 0.0)
        # if True, interpolate from the measured spawn pose to default over warmup_sec (no violent snap);
        # if False, command the default pose immediately.
        self.declare_parameter('warmup_interpolate', True)
        # nav_msgs/Odometry twist convention. REP-103 says twist is in the child (body) frame -> True.
        # If Isaac publishes the base velocity in the world frame, set this False to rotate it in.
        self.declare_parameter('odom_twist_in_body_frame', True)
        # topic names (override to match your Isaac Sim action graphs)
        self.declare_parameter('joint_states_topic', 'joint_states')
        self.declare_parameter('imu_topic', 'imu')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('joint_command_topic', 'joint_command')
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)]
        )

        self._logger = self.get_logger()

        self._decimation = int(self.get_parameter('decimation').value)
        self._odom_in_body = bool(self.get_parameter('odom_twist_in_body_frame').value)
        self._warmup_sec = float(self.get_parameter('warmup_sec').value)
        self._warmup_interp = bool(self.get_parameter('warmup_interpolate').value)

        # ---- policy + io descriptors ----------------------------------------------------------------
        self.policy_path = self.get_parameter('policy_path').value
        self._load_policy()
        self._load_io_descriptors()

        # ---- QoS / IO -------------------------------------------------------------------------------
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        self._joint_publisher = self.create_publisher(
            JointState, self.get_parameter('joint_command_topic').value, qos_profile=sim_qos_profile)

        # async caches (slow-changing / command inputs — no need to time-sync these)
        self._cmd_vel = Twist()
        self._odom_lin_vel_w = np.zeros(3)  # base linear velocity as reported by /odom
        self.create_subscription(
            Twist, self.get_parameter('cmd_vel_topic').value, self._cmd_vel_callback, qos_profile=10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._odom_callback, qos_profile=sim_qos_profile)

        # joint_states + imu must be temporally aligned -> synchronize them
        self._joint_states_sub = Subscriber(
            self, JointState, self.get_parameter('joint_states_topic').value, qos_profile=sim_qos_profile)
        self._imu_sub = Subscriber(
            self, Imu, self.get_parameter('imu_topic').value, qos_profile=sim_qos_profile)
        self.sync = TimeSynchronizer([self._joint_states_sub, self._imu_sub], 10)
        self.sync.registerCallback(self._tick)

        # ---- runtime state --------------------------------------------------------------------------
        self._joint_command = JointState()
        self._previous_action = np.zeros(self.num_joints)
        self.action = np.zeros(self.num_joints)
        self._policy_counter = 0
        # warmup state: ease the robot into the default pose before engaging the policy
        self._first_tick_time = None
        self._warmup_start_pos = None
        self._warmed_up = self._warmup_sec <= 0.0

        self._logger.info(
            f"FullbodyController ready: {self.num_joints} joints, obs_dim={self.obs_dim}, "
            f"action_scale={self.action_scale}, decimation={self._decimation}, "
            f"warmup_sec={self._warmup_sec}")

    # ---- loading -----------------------------------------------------------------------------------

    def _load_policy(self):
        """Load the TorchScript policy from disk."""
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.policy = torch.jit.load(buffer, map_location='cpu')
        self.policy.eval()

    def _load_io_descriptors(self):
        """Load joint order / defaults / scaling / obs layout from Isaac Lab's IO_descriptors.yaml.

        Expects the layout produced by ``scripts/environments/export_IODescriptors.py``:
        a single ``actions`` entry (JointPositionAction) plus an ``observations.policy`` list
        whose ``shape`` entries sum to the policy's observation dimension.
        """
        desc_path = self.get_parameter('io_descriptors_path').value
        if not desc_path:
            desc_path = os.path.join(os.path.dirname(self.policy_path), 'IO_descriptors.yaml')
        if not os.path.isfile(desc_path):
            raise FileNotFoundError(
                f"IO_descriptors.yaml not found at '{desc_path}'. Re-export the policy with "
                "export_io_descriptors=True (or run scripts/environments/export_IODescriptors.py) "
                "and place the file next to the policy, or set the 'io_descriptors_path' parameter.")

        with open(desc_path, 'r') as f:
            desc = yaml.safe_load(f)

        # ---- action term ----
        actions = desc.get('actions') or []
        if not actions:
            raise ValueError(f"No 'actions' entries found in {desc_path}.")
        # The deployed policy is single-headed: one joint-position action term.
        action = actions[0]
        self.joint_names = list(action['joint_names'])
        self.num_joints = len(self.joint_names)
        self.default_pos = np.array(action['offset'], dtype=np.float64)
        self.action_scale = float(action['scale'])

        # ---- observation layout ----
        # observations is a dict keyed by group name; the policy group is what the actor consumes.
        obs_groups = desc.get('observations') or {}
        policy_obs = obs_groups.get('policy')
        if not policy_obs:
            raise ValueError(f"No 'observations.policy' group found in {desc_path}.")
        self.obs_dim = sum(int(term['shape'][0]) for term in policy_obs)

        # map a joint name -> its index in the policy's observation/action order
        self._name_to_policy_idx = {name: i for i, name in enumerate(self.joint_names)}
        self._logger.info(f"Loaded IO descriptors from {desc_path}")

    # ---- callbacks ---------------------------------------------------------------------------------

    def _cmd_vel_callback(self, msg: Twist):
        """Cache the latest velocity command."""
        self._cmd_vel = msg

    def _odom_callback(self, msg: Odometry):
        """Cache the latest base linear velocity from odometry."""
        v = msg.twist.twist.linear
        self._odom_lin_vel_w = np.array([v.x, v.y, v.z])

    def _tick(self, joint_state: JointState, imu: Imu):
        """Run one control step from synchronized joint state + IMU.

        On startup, hold/ease the robot into the policy's default joint pose for ``warmup_sec``
        so the first real observation is in-distribution, then engage the policy.
        """
        now = self.get_clock().now()
        if self._first_tick_time is None:
            self._first_tick_time = now
            self._warmup_start_pos = self._map_joint_pos(joint_state)

        if not self._warmed_up:
            elapsed = (now - self._first_tick_time).nanoseconds * 1e-9
            if elapsed < self._warmup_sec:
                # hold last_action at zero so the policy's first obs sees a clean reset state
                self.action = np.zeros(self.num_joints)
                self._previous_action = np.zeros(self.num_joints)
                if self._warmup_interp and self._warmup_start_pos is not None:
                    alpha = min(elapsed / self._warmup_sec, 1.0)
                    target = (1.0 - alpha) * self._warmup_start_pos + alpha * self.default_pos
                else:
                    target = self.default_pos
                self._publish(target)
                return
            self._warmed_up = True
            self._logger.info("Warmup complete; engaging policy.")

        self.forward(joint_state, imu)
        # JointPositionActionCfg: target = default + scale * action
        self._publish(self.default_pos + self.action * self.action_scale)

    def _publish(self, positions: np.ndarray):
        """Publish a 37-joint position target in policy order."""
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names
        self._joint_command.position = positions.tolist()
        self._joint_command.velocity = np.zeros(self.num_joints).tolist()
        self._joint_command.effort = np.zeros(self.num_joints).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _map_joint_pos(self, joint_state: JointState) -> np.ndarray:
        """Map an incoming ``joint_states`` message into policy joint order (by name)."""
        joint_pos = self.default_pos.copy()
        for src_idx, name in enumerate(joint_state.name):
            dst = self._name_to_policy_idx.get(name)
            if dst is not None:
                joint_pos[dst] = joint_state.position[src_idx]
        return joint_pos

    # ---- policy ------------------------------------------------------------------------------------

    def _compute_observation(self, joint_state: JointState, imu: Imu) -> np.ndarray:
        """Assemble the observation in the exact training order (body frame)."""
        # body orientation: quat_to_rot gives R_WB (body->world); transpose -> R_BW (world->body)
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BW = self.quat_to_rot_matrix(quat_array).T

        # base linear velocity (body frame)
        if self._odom_in_body:
            lin_vel_b = self._odom_lin_vel_w  # already body frame (REP-103)
        else:
            lin_vel_b = np.matmul(R_BW, self._odom_lin_vel_w)  # rotate world -> body

        ang_vel_b = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
        gravity_b = np.matmul(R_BW, np.array([0.0, 0.0, -1.0]))
        cmd_vel = np.array([self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z])

        # map incoming joint_states (any order) into the policy's joint order
        joint_pos = self._map_joint_pos(joint_state)
        joint_vel = np.zeros(self.num_joints)
        for src_idx, name in enumerate(joint_state.name):
            dst = self._name_to_policy_idx.get(name)
            if dst is not None and src_idx < len(joint_state.velocity):
                joint_vel[dst] = joint_state.velocity[src_idx]

        obs = np.zeros(self.obs_dim)
        n = self.num_joints
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = cmd_vel
        obs[12:12 + n] = joint_pos - self.default_pos
        obs[12 + n:12 + 2 * n] = joint_vel
        obs[12 + 2 * n:12 + 3 * n] = self._previous_action
        return obs

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Run the policy network."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_t).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Compute a new action every ``decimation`` ticks; hold it otherwise."""
        obs = self._compute_observation(joint_state, imu)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    # ---- utils -------------------------------------------------------------------------------------

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix (body -> world)."""
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def _get_stamp_prefix(self) -> str:
        """Timestamp prefix for logging."""
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'


def main(args=None):
    """Initialize and spin the G1 controller node."""
    rclpy.init(args=args)
    node = FullbodyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
