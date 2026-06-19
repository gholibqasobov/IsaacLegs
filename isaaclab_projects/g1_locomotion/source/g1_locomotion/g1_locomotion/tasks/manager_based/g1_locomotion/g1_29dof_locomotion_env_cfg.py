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
# Action settings
##


@configclass
class G1_29DOF_ActionsCfg:
    """Split actions so the legs keep full authority while the upper body (the 3 waist
    DOFs + arms/wrists) is given a smaller scale, limiting torso/arm sway during
    locomotion. The two terms together cover all 29 joints with no overlap."""

    legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],  # 12 joints
        scale=0.5,
        use_default_offset=True,
    )
    upper_body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_.*_joint", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],  # 17 joints
        scale=0.25,  # half the leg scale -> limited waist/arm travel
        use_default_offset=True,
    )


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
        weight=-0.25,
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
    # Penalized strongly to keep the upper body from swaying (the roll/pitch DOFs are what
    # let the torso oscillate); tune toward -2.0 if sway remains.
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )

    # Keep the body slightly lower (forces a slight knee bend) for a more stable stance.
    # The robot otherwise locks its legs straight to minimize hip/knee torque. Pelvis init
    # height is 0.74; target just below it. Flat terrain -> no sensor_cfg. Tune: raise
    # magnitude / lower target if still too tall; reduce if it squats / gait degrades.
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.72},
    )


##
# Environment configuration
##


@configclass
class G1_29DOF_LocomotionEnvCfg(G1LocomotionEnvCfg):
    """Locomotion env for the 29-DOF G1 import."""

    scene: G1_29DOF_LocomotionSceneCfg = G1_29DOF_LocomotionSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: G1_29DOF_ActionsCfg = G1_29DOF_ActionsCfg()
    rewards: G1_29DOF_RewardsCfg = G1_29DOF_RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # The base env weights feet_air_time at 0.75 (3x the official G1 reference of 0.25),
        # which over-rewards long single-stance swing time and lets the policy settle into a
        # limping gait with one slow, high, fast-snapping leg. Match the proven reference.
        self.rewards.feet_air_time.weight = 0.25
