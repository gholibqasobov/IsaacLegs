#!/usr/bin/env python3
"""Configure a robot's joint drives for policy deployment from the trained YAMLs.

Run this inside Isaac Sim (Script Editor or an extension) after the robot is on
the stage. It replaces hardcoded kp/kd/effort numbers by reading them straight
from the files Isaac Lab exported alongside the policy:

  * ``IO_descriptors.yaml`` -> per-joint ``default_joint_stiffness`` (kp),
    ``default_joint_damping`` (kd) and ``default_joint_armature`` (already
    flattened, in ``joint_names`` order).
  * ``env.yaml`` -> ``scene.robot.actuators.*.effort_limit_sim`` (effort limit),
    which is NOT present in IO_descriptors and must be expanded from the
    actuator group regexes.

Everything is mapped by joint NAME, because the articulation's DOF order
(``art.dof_names``) generally differs from the YAML ``joint_names`` order.

It also (optionally) applies the USD schemas equivalent to the Stage panel's
  Add -> Physics -> Angular Drive        (UsdPhysics.DriveAPI, "angular")
  Add -> Physics -> Joint State (angular) (PhysxSchema.JointStateAPI, "angular")
so that ``set_gains(save_to_usd=True)`` has somewhere to write. The numeric
gains are written by Isaac core's ``set_gains`` (SI/radians); we never hand-author
the per-degree PhysX drive values.

Usage (Isaac Sim Script Editor)::

    # G1 (defaults, no args needed)
    from setup_robot_gains import main; main()

    # Go2 (or any other robot whose policy is shipped in the same package)
    import os
    from setup_robot_gains import main, _REPO_ROOT
    main(
        robot_prim="/World/go2",
        policy_dir=os.path.join(
            _REPO_ROOT, "src", "fullbody_controller", "policy", "go2_locomotion"),
    )

Usage (CLI, e.g. headless / launch script)::

    python setup_robot_gains.py \\
        --robot-prim /World/go2 \\
        --policy-dir src/fullbody_controller/policy/go2_locomotion
"""

import os
import re

import numpy as np
import yaml
from isaacsim.core.prims import SingleArticulation

# ---- defaults (G1) -----------------------------------------------------------
# These keep the script a no-arg ``main()`` call for the original G1 use case.
# G1_POLICY_DIR is resolved relative to this file so the repo is portable
# (script at <repo>/isaacsim_envs/gui/, policies at <repo>/src/fullbody_controller/policy/).
G1_ROBOT_PRIM = "/World/g1"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G1_POLICY_DIR = os.path.join(
    _REPO_ROOT, "src", "fullbody_controller", "policy", "g1_locomotion")


class _IsaacLabLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the Python tags Isaac Lab writes into env.yaml.

    Isaac Lab dumps configs with ``!!python/tuple``, ``!!python/object/apply:...``
    (e.g. ``builtins.slice``) and similar tags that ``yaml.safe_load`` rejects.
    We only read plain data (the ``actuators`` section), so convert tuples to
    lists and drop every other Python-constructed object as ``None``.
    """


def _construct_tuple(loader, node):
    """Construct ``!!python/tuple`` nodes as plain lists."""
    return loader.construct_sequence(node)


def _ignore_python_object(loader, tag_suffix, node):
    """Drop any other ``!!python/...`` object (slices, enums, callables) as None."""
    return None


_IsaacLabLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
_IsaacLabLoader.add_multi_constructor("tag:yaml.org,2002:python/", _ignore_python_object)


def _apply_angular_schemas(robot_prim_path: str, default_pos_rad: dict) -> None:
    """Apply angular DriveAPI + JointStateAPI to every revolute joint and seed the init pose.

    This is the scripted equivalent of the Stage panel context-menu items
    'Add -> Physics -> Angular Drive' and 'Add -> Physics -> Joint State (angular)'.
    On the stock G1 USD these usually already exist, so this acts as a safety net.

    ``default_pos_rad`` maps joint name -> default position in RADIANS (Isaac Lab
    convention). USD angular joints are authored in DEGREES, so the value is
    converted before it is written to the drive target and the joint state.
    """
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):  # angular joints only
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            pos_rad = default_pos_rad.get(prim.GetName())
            if pos_rad is not None:
                pos_deg = float(np.degrees(pos_rad))  # rad -> deg for USD angular joints
                drive.CreateTargetPositionAttr().Set(pos_deg)
                state.CreatePositionAttr().Set(pos_deg)


def main(
    robot_prim: str = G1_ROBOT_PRIM,
    policy_dir: str = G1_POLICY_DIR,
    apply_physics_schemas: bool = True,
) -> None:
    """Read the YAMLs and write per-joint gains/efforts/armature onto the robot.

    Args:
        robot_prim: USD path to the articulation root (e.g. ``/World/g1``,
            ``/World/go2``). Check the Stage panel to confirm.
        policy_dir: Directory containing ``IO_descriptors.yaml`` and
            ``env.yaml`` (the pair Isaac Lab exports next to ``policy.pt``).
        apply_physics_schemas: If True, also apply the angular DriveAPI /
            JointStateAPI schemas to every revolute joint and seed the init
            pose. Safe no-op on USDs where those schemas already exist.
    """
    io_desc_path = os.path.join(policy_dir, "IO_descriptors.yaml")
    env_yaml_path = os.path.join(policy_dir, "env.yaml")

    # ---- per-joint kp/kd/armature (already flattened, in joint_names order) ----
    with open(io_desc_path) as f:
        desc = yaml.safe_load(f)
    robot = desc["articulations"]["robot"]
    yaml_names = robot["joint_names"]
    kp_by_name = dict(zip(yaml_names, robot["default_joint_stiffness"]))
    kd_by_name = dict(zip(yaml_names, robot["default_joint_damping"]))
    arm_by_name = dict(zip(yaml_names, robot["default_joint_armature"]))
    pos_by_name = dict(zip(yaml_names, robot["default_joint_pos"]))  # radians (Isaac Lab)

    # ---- effort limits from env.yaml actuator groups (regex -> joints) ----
    with open(env_yaml_path) as f:
        env = yaml.load(f, Loader=_IsaacLabLoader)
    actuators = env["scene"]["robot"]["actuators"]
    eff_by_name = {}
    for group in actuators.values():
        eff = float(group["effort_limit_sim"])
        for pattern in group["joint_names_expr"]:
            for n in yaml_names:
                if re.fullmatch(pattern, n):
                    eff_by_name[n] = eff

    # ---- optionally add the angular Drive / JointState schemas + init pose ----
    if apply_physics_schemas:
        _apply_angular_schemas(robot_prim, pos_by_name)

    # ---- build arrays in the articulation's ACTUAL dof order (map by name!) ----
    art = SingleArticulation(robot_prim)
    art.initialize()
    names = art.dof_names

    missing = [n for n in names if n not in kp_by_name or n not in eff_by_name]
    if missing:
        raise RuntimeError(f"joints in articulation not found in YAML: {missing}")

    kps = np.array([kp_by_name[n] for n in names], dtype=np.float32)
    kds = np.array([kd_by_name[n] for n in names], dtype=np.float32)
    arms = np.array([arm_by_name[n] for n in names], dtype=np.float32)
    effs = np.array([eff_by_name[n] for n in names], dtype=np.float32)

    art.get_articulation_controller().set_gains(kps=kps, kds=kds, save_to_usd=True)
    art._articulation_view.set_max_efforts(effs)  # on the _articulation_view
    art._articulation_view.set_armatures(arms.reshape(1, -1))

    for n, a, b, c, d in zip(names, kps, kds, effs, arms):
        pos_deg = np.degrees(pos_by_name[n])
        print(f"{n:28s} kp={a:6.1f} kd={b:5.1f} eff={c:6.1f} arm={d:.3f} init={pos_deg:7.2f}deg")
    print("DONE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-prim", default=G1_ROBOT_PRIM,
                        help=f"articulation-root prim path (default: {G1_ROBOT_PRIM})")
    parser.add_argument("--policy-dir", default=G1_POLICY_DIR,
                        help="directory containing IO_descriptors.yaml + env.yaml "
                             f"(default: {G1_POLICY_DIR})")
    parser.add_argument("--no-physics-schemas", action="store_true",
                        help="skip applying angular Drive / JointState schemas")
    args = parser.parse_args()
    main(
        robot_prim=args.robot_prim,
        policy_dir=args.policy_dir,
        apply_physics_schemas=not args.no_physics_schemas,
    )
