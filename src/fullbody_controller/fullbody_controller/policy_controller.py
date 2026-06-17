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

"""ROS 2 controller that runs a trained Isaac Lab policy.

* control rate = 50 Hz.

The exact joint order, default positions, action scaling and per-term observation layout
are loaded from Isaac Lab's ``IO_descriptors.yaml`` (produced when the env is trained or
exported with ``export_io_descriptors=True``), so this node is robot-agnostic for any
policy whose observation terms are listed in ``OBS_PRODUCERS`` below.
"""

import io
import os
import time
from types import SimpleNamespace

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

    # ---- observation producer registry ---------------------------------------------------------
    # Maps an IO-descriptor term name -> the name of an instance method that returns that term's
    # 1-D slice of the observation vector. The dispatch loop in ``_compute_observation`` walks
    # ``self.obs_terms`` in order, calls the producer for each, and concatenates.
    #
    # To add a new obs term:
    #   1. Write an ``_obs_<name>(self, ctx)`` method that returns a 1-D np.ndarray.
    #   2. Add the ``name -> method_name`` mapping below.
    #   3. If the term needs a NEW topic, add a matching entry to TOPIC_SOURCES (see below).
    # Unknown term names fail fast at startup with the available registry keys in the error.
    OBS_PRODUCERS = {
        "base_lin_vel":       "_obs_base_lin_vel",
        "base_ang_vel":       "_obs_base_ang_vel",
        "projected_gravity":  "_obs_projected_gravity",
        "generated_commands": "_obs_generated_commands",
        "joint_pos_rel":      "_obs_joint_pos_rel",
        "joint_vel_rel":      "_obs_joint_vel_rel",
        "last_action":        "_obs_last_action",
    }

    # ---- topic source registry ----------------------------------------------------------------
    # Declarative description of every ROS topic the controller can SUBSCRIBE to. ``__init__``
    # walks this list to declare launch parameters, build the TimeSynchronizer that drives the
    # control loop, and create the async subscriptions that feed obs producer caches.
    #
    # Async sources can be gated on which observation terms are present in the loaded IO
    # descriptor (see ``feeds_terms``), so the controller stays silent about sensors a given
    # policy does not consume.
    #
    # Fields:
    #   "key"          str   - short identifier (also used in log messages)
    #   "param"        str   - ROS parameter name for the topic (override with -p <param>:=...)
    #   "default"      str   - default topic name; '' means "must be supplied by the user"
    #   "msg_type"     class - ROS message type
    #   "mode"         str   - "tick"  : TimeSynchronizer'd input driving ``_tick``. Only the
    #                                    canonical joint_states + imu pair use this mode;
    #                                    adding a third tick source requires editing _tick.
    #                          "async" : cached by the named callback method, read by producers.
    #   "callback"     str   - bound-method name handling each message (async only)
    #   "qos"          str   - "sim" (RELIABLE/VOLATILE/KEEP_ALL for Isaac Sim) or
    #                          "default" (depth-10 system default; good for low-rate commands)
    #   "feeds_terms"  None  - always required (subscription always created), OR
    #                  tuple - obs term names this source enables; subscription is skipped if
    #                          none of these terms appear in the loaded descriptor.
    #
    # ---- TODO: to add a new sensor topic, copy the commented LiDAR template below ----
    # Steps:
    #   1. Import the message type at the top of this file.
    #   2. Add an entry to TOPIC_SOURCES here (use a tuple in feeds_terms to enable gating).
    #   3. Add a cache attribute (e.g. ``self._lidar_scan = None``) in ``_init_topic_caches``.
    #   4. Add the callback method on this class (e.g. ``_lidar_callback``).
    #   5. Add the obs producer method and an OBS_PRODUCERS entry (see above).
    #   6. Launch with the topic name, e.g. ``ros2 launch ... lidar_topic:=/scan``.
    TOPIC_SOURCES = [
        {"key": "joint_states", "param": "joint_states_topic", "default": "joint_states",
         "msg_type": JointState, "mode": "tick",  "qos": "sim",
         "feeds_terms": None},
        {"key": "imu",          "param": "imu_topic",          "default": "imu",
         "msg_type": Imu,        "mode": "tick",  "qos": "sim",
         "feeds_terms": None},
        {"key": "odom",         "param": "odom_topic",         "default": "odom",
         "msg_type": Odometry,   "mode": "async", "callback": "_odom_callback",
         "qos": "sim",     "feeds_terms": None},
        {"key": "cmd_vel",      "param": "cmd_vel_topic",      "default": "cmd_vel",
         "msg_type": Twist,      "mode": "async", "callback": "_cmd_vel_callback",
         "qos": "default", "feeds_terms": None},
        # TODO -- LiDAR extension template. Uncomment + complete steps 1, 3, 4, 5 above.
        # {"key": "lidar",      "param": "lidar_topic",        "default": "",
        #  "msg_type": LaserScan, "mode": "async", "callback": "_lidar_callback",
        #  "qos": "default", "feeds_terms": ("lidar_scan",)},
    ]

    # ---- publisher registry -------------------------------------------------------------------
    # Every topic the controller PUBLISHES to. ``__init__`` creates each publisher and stores it
    # on the named attribute, so the runtime code can ``self._joint_publisher.publish(msg)``.
    PUBLISHERS = [
        {"key": "joint_command", "param": "joint_command_topic", "default": "joint_command",
         "msg_type": JointState, "attr": "_joint_publisher", "qos": "sim"},
    ]

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
        # topic-name parameters are derived from TOPIC_SOURCES + PUBLISHERS so adding a new sensor
        # is a one-line registry edit rather than a copy/paste here.
        for entry in self.TOPIC_SOURCES + self.PUBLISHERS:
            self.declare_parameter(entry['param'], entry['default'])
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
        # qos label -> concrete QoS used by both registries
        qos_map = {"sim": sim_qos_profile, "default": 10}

        # ---- publishers (from PUBLISHERS registry) ---------------------------------------------------
        for entry in self.PUBLISHERS:
            topic = self.get_parameter(entry['param']).value
            pub = self.create_publisher(entry['msg_type'], topic, qos_profile=qos_map[entry['qos']])
            setattr(self, entry['attr'], pub)

        # ---- topic source caches (seeded so producers do not see NoneType before first message) ----
        self._init_topic_caches()

        # ---- subscriptions (from TOPIC_SOURCES registry) --------------------------------------------
        # Walk the registry: 'tick' sources go into one TimeSynchronizer driving the control loop;
        # 'async' sources get a normal subscription that writes to a cache. Async sources whose
        # ``feeds_terms`` tuple is set are SKIPPED when none of those obs terms appear in the
        # loaded descriptor -- so adding optional sensors does not cost anything until a policy
        # actually consumes them.
        tick_subs = []
        for entry in self.TOPIC_SOURCES:
            if entry['mode'] == 'async' and entry['feeds_terms'] is not None:
                if not any(t in self._obs_term_names for t in entry['feeds_terms']):
                    self._logger.info(
                        f"skipping subscription for '{entry['key']}': descriptor has no terms "
                        f"in {tuple(entry['feeds_terms'])}.")
                    continue
            topic = self.get_parameter(entry['param']).value
            if not topic:
                detail = (f" (required by descriptor terms {entry['feeds_terms']})"
                          if entry['feeds_terms'] is not None else " (always required)")
                raise RuntimeError(
                    f"topic source '{entry['key']}' is enabled but parameter '{entry['param']}' "
                    f"is empty{detail}. Pass a topic name at launch, e.g. "
                    f"-p {entry['param']}:=/your/topic.")
            qos = qos_map[entry['qos']]
            if entry['mode'] == 'tick':
                sub = Subscriber(self, entry['msg_type'], topic, qos_profile=qos)
                tick_subs.append(sub)
                setattr(self, f"_{entry['key']}_sub", sub)
            else:  # async
                cb = getattr(self, entry['callback'])
                self.create_subscription(entry['msg_type'], topic, cb, qos_profile=qos)

        if not tick_subs:
            raise RuntimeError(
                "No 'tick' mode entries in TOPIC_SOURCES: the controller has no input to drive "
                "the control loop and will never run.")
        # NOTE: _tick's positional args bind to tick_subs in registry order (joint_states, imu).
        # If you add a third tick source, update _tick's signature.
        self.sync = TimeSynchronizer(tick_subs, 10)
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

        self.obs_terms = []
        for term in policy_obs:
            name = term['name']
            shape = list(term['shape'])
            history_length = int((term.get('overloads') or {}).get('history_length', 0))
            if history_length > 0:
                raise NotImplementedError(
                    f"observation term '{name}' uses history_length={history_length}; this "
                    "controller does not implement history buffering yet. Add a ring buffer "
                    "on self and produce the flattened history from the producer to support it.")
            if name not in self.OBS_PRODUCERS:
                raise RuntimeError(
                    f"No observation producer registered for term '{name}' "
                    f"(descriptor: {desc_path}). Known terms: {sorted(self.OBS_PRODUCERS)}. "
                    "Add a producer method and an OBS_PRODUCERS entry, or fix the descriptor.")
            self.obs_terms.append({'name': name, 'shape': shape, 'history_length': history_length})

        self._obs_term_names = {t['name'] for t in self.obs_terms}
        self.obs_dim = sum(int(t['shape'][0]) for t in self.obs_terms)

        # map a joint name -> its index in the policy's observation/action order
        self._name_to_policy_idx = {name: i for i, name in enumerate(self.joint_names)}
        self._logger.info(
            f"Loaded IO descriptors from {desc_path}: "
            f"obs terms = {[t['name'] for t in self.obs_terms]}")

    # ---- topic source caches -----------------------------------------------------------------------

    def _init_topic_caches(self):
        """Seed cache attributes that obs producers may read before the first message arrives.

        Add one line here for every async ``TOPIC_SOURCES`` entry whose callback writes to
        ``self.<attr>``. Initialising up-front avoids ``AttributeError`` on the first tick if
        the policy starts running before the sensor publishes.
        """
        self._cmd_vel = Twist()
        self._odom_lin_vel_w = np.zeros(3)  # base linear velocity as reported by /odom
        # Extension caches (initialise alongside each new TOPIC_SOURCES entry):
        # self._lidar_scan = None

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

        The argument list is positional and bound to the order of ``TOPIC_SOURCES`` entries
        with ``mode='tick'`` (today: joint_states, then imu). If a third tick source is added,
        update this signature.

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
        """Publish a joint-position target in the policy's joint order."""
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

    def _build_obs_ctx(self, joint_state: JointState, imu: Imu) -> SimpleNamespace:
        """Pre-compute everything the observation producers need from the sync'd inputs."""
        # body orientation: quat_to_rot gives R_WB (body->world); transpose -> R_BW (world->body)
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BW = self.quat_to_rot_matrix(quat_array).T

        # base linear velocity (body frame)
        if self._odom_in_body:
            lin_vel_b = self._odom_lin_vel_w  # already body frame (REP-103)
        else:
            lin_vel_b = np.matmul(R_BW, self._odom_lin_vel_w)  # rotate world -> body

        ang_vel_b = np.array(
            [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
        gravity_b = np.matmul(R_BW, np.array([0.0, 0.0, -1.0]))
        cmd_vel = np.array(
            [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z])

        # map incoming joint_states (any order) into the policy's joint order
        joint_pos = self._map_joint_pos(joint_state)
        joint_vel = np.zeros(self.num_joints)
        for src_idx, name in enumerate(joint_state.name):
            dst = self._name_to_policy_idx.get(name)
            if dst is not None and src_idx < len(joint_state.velocity):
                joint_vel[dst] = joint_state.velocity[src_idx]

        return SimpleNamespace(
            R_BW=R_BW, lin_vel_b=lin_vel_b, ang_vel_b=ang_vel_b, gravity_b=gravity_b,
            cmd_vel=cmd_vel, joint_pos=joint_pos, joint_vel=joint_vel,
        )

    def _compute_observation(self, joint_state: JointState, imu: Imu) -> np.ndarray:
        """Assemble the observation in the descriptor's term order via OBS_PRODUCERS."""
        ctx = self._build_obs_ctx(joint_state, imu)
        parts = []
        for term in self.obs_terms:
            method = getattr(self, self.OBS_PRODUCERS[term['name']])
            part = np.asarray(method(ctx), dtype=np.float64).reshape(-1)
            expected = int(term['shape'][0])
            if part.size != expected:
                raise RuntimeError(
                    f"observation producer '{term['name']}' returned {part.size} dims, "
                    f"descriptor expected {expected}.")
            parts.append(part)
        return np.concatenate(parts) if parts else np.zeros(0)

    # ---- observation producers (one per OBS_PRODUCERS entry) ---------------------------------------

    def _obs_base_lin_vel(self, ctx):
        return ctx.lin_vel_b

    def _obs_base_ang_vel(self, ctx):
        return ctx.ang_vel_b

    def _obs_projected_gravity(self, ctx):
        return ctx.gravity_b

    def _obs_generated_commands(self, ctx):
        return ctx.cmd_vel

    def _obs_joint_pos_rel(self, ctx):
        return ctx.joint_pos - self.default_pos

    def _obs_joint_vel_rel(self, ctx):
        return ctx.joint_vel

    def _obs_last_action(self, ctx):
        return self._previous_action

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
