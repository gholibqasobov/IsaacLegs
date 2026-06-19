# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the 29-DOF Unitree G1 locomotion policy.

A humanoid has a single sagittal symmetry plane, so we augment each sample with its
left-right mirror (2x augmentation). Mirroring requires, for every joint-indexed quantity,
swapping each left joint with its right partner (and vice-versa) and flipping the sign of
the roll/yaw joints (rotations about the forward/vertical axes flip under a left-right
mirror; pitch/knee/elbow rotations about the lateral axis are preserved).

The index maps are built **programmatically** from the live articulation joint ordering and
the action-term grouping (legs then upper body), so they stay correct regardless of the
exact joint order Isaac Sim assigns or future joint-set tweaks. They are computed once and
cached on the environment instance.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Cache attribute name on the env instance.
_MAPS_ATTR = "_g1_29dof_symmetry_maps"

# Joint-name substrings that belong to the "legs" action term (must match the
# joint_names regexes of ``G1_29DOF_ActionsCfg.legs``). Everything else is "upper body".
_LEG_KEYWORDS = ("hip", "knee", "ankle")


def _mirror_name(name: str) -> str:
    """Return the left/right-mirrored joint name (centerline joints map to themselves)."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name  # waist_* and any other centerline joint


def _mirror_sign(name: str) -> float:
    """Sign applied to a joint value under a left-right mirror.

    Roll (about forward x) and yaw (about vertical z) flip; pitch (about lateral y),
    knee and elbow are preserved. Validated by the symmetric rest pose
    (``left_shoulder_roll = +0.16`` / ``right_shoulder_roll = -0.16``).
    """
    return -1.0 if ("roll" in name or "yaw" in name) else 1.0


def _build_maps(env: ManagerBasedRLEnv) -> dict:
    """Build and cache the permutation/sign tensors for the mirror transform."""
    device = env.device
    names = list(env.scene["robot"].joint_names)  # articulation order
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # --- articulation-order maps (used by joint_pos / joint_vel obs segments) ---
    art_perm = torch.tensor([name_to_idx[_mirror_name(name)] for name in names], device=device)
    art_sign = torch.tensor([_mirror_sign(name) for name in names], device=device)

    # --- action-order maps (legs term then upper-body term, each in articulation order) ---
    # Reproduces the concatenation order of G1_29DOF_ActionsCfg (legs, then upper_body).
    legs = [i for i, name in enumerate(names) if any(k in name for k in _LEG_KEYWORDS)]
    upper = [i for i, name in enumerate(names) if not any(k in name for k in _LEG_KEYWORDS)]
    act_to_art = legs + upper  # action position -> articulation index
    art_to_act = {art: pos for pos, art in enumerate(act_to_art)}
    # For each action position p (art joint a), the mirror lives at action position of art_perm[a].
    act_perm = torch.tensor(
        [art_to_act[int(art_perm[act_to_art[p]])] for p in range(n)], device=device
    )
    act_sign = torch.tensor([_mirror_sign(names[act_to_art[p]]) for p in range(n)], device=device)

    return {
        "art_perm": art_perm,
        "art_sign": art_sign,
        "act_perm": act_perm,
        "act_sign": act_sign,
    }


def _get_maps(env: ManagerBasedRLEnv) -> dict:
    maps = getattr(env, _MAPS_ATTR, None)
    if maps is None:
        maps = _build_maps(env)
        setattr(env, _MAPS_ATTR, maps)
    return maps


def _mirror_joint_segment(data: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Swap left<->right joints and flip roll/yaw signs for a (batch, n) joint segment."""
    return data[:, perm] * sign


def _mirror_policy_obs(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply the left-right mirror to the 99-dim policy observation vector.

    Layout: base_lin_vel[0:3], base_ang_vel[3:6], projected_gravity[6:9],
    velocity_command[9:12], joint_pos[12:41], joint_vel[41:70], last_action[70:99].
    """
    maps = _get_maps(env)
    out = obs.clone()
    device = obs.device
    # base linear velocity (vx, vy, vz) -> flip y
    out[:, 0:3] = obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0], device=device)
    # base angular velocity (wx, wy, wz) -> flip x (roll) and z (yaw)
    out[:, 3:6] = obs[:, 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    # projected gravity (gx, gy, gz) -> flip y
    out[:, 6:9] = obs[:, 6:9] * torch.tensor([1.0, -1.0, 1.0], device=device)
    # velocity command (vx, vy, wz) -> flip lateral vel and yaw rate
    out[:, 9:12] = obs[:, 9:12] * torch.tensor([1.0, -1.0, -1.0], device=device)
    # joint pos / joint vel are in articulation order
    out[:, 12:41] = _mirror_joint_segment(obs[:, 12:41], maps["art_perm"], maps["art_sign"])
    out[:, 41:70] = _mirror_joint_segment(obs[:, 41:70], maps["art_perm"], maps["art_sign"])
    # last action is in action order (legs term then upper-body term)
    out[:, 70:99] = _mirror_joint_segment(obs[:, 70:99], maps["act_perm"], maps["act_sign"])
    return out


def _mirror_actions(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    """Apply the left-right mirror to the action vector (action-term order)."""
    maps = _get_maps(env)
    return _mirror_joint_segment(actions.clone(), maps["act_perm"], maps["act_sign"])


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with their left-right mirror (2x augmentation).

    Signature matches ``RslRlSymmetryCfg.data_augmentation_func``. Returns
    ``(obs_aug, actions_aug)`` where each is the original batch followed by its mirror,
    or ``None`` for whichever input was ``None``.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_actions(env.unwrapped, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
