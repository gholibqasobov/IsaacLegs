# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environment for the custom-imported Unitree G1 29-DOF robot.

This reuses :class:`G1LocomotionEnvCfg` as the base and overrides only what differs
for the 29-DOF URDF import:
  * the robot articulation (``G1_29DOF_CFG``),
  * the reward terms whose joint names changed (arms, waist) or no longer exist
    (fingers), since the 29-DOF URDF has no hands.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from g1_locomotion.robots.unitree_g1 import G1_29DOF_CFG  # isort:skip

from . import mdp
from .g1_locomotion_env_cfg import G1LocomotionEnvCfg, G1LocomotionSceneCfg, RewardsCfg


##
# Scene definition
##


@configclass
class G1_29DOF_LocomotionSceneCfg(G1LocomotionSceneCfg):
    """Scene that swaps in the 29-DOF G1 articulation."""

    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# MDP settings
##


@configclass
class G1_29DOF_RewardsCfg(RewardsCfg):
    """Rewards adjusted for the 29-DOF joint set."""

    # 29-DOF arms have a single elbow joint and three wrist joints (the base config's
    # ``.*_elbow_pitch_joint`` / ``.*_elbow_roll_joint`` do not exist here).
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )

    # The 29-DOF URDF has no hand/finger joints, so disable the finger deviation term.
    joint_deviation_fingers = None

    # The waist is three separate joints (yaw/roll/pitch) instead of a single torso_joint.
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )


##
# Environment configuration
##


@configclass
class G1_29DOF_LocomotionEnvCfg(G1LocomotionEnvCfg):
    """Locomotion env for the 29-DOF G1 import."""

    scene: G1_29DOF_LocomotionSceneCfg = G1_29DOF_LocomotionSceneCfg(num_envs=4096, env_spacing=4.0)
    rewards: G1_29DOF_RewardsCfg = G1_29DOF_RewardsCfg()
