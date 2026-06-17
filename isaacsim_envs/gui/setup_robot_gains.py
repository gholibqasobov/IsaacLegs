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

It also (optionally) applies the USD schemas equivalent to the Stage panel's
  Add -> Physics -> Angular Drive        (UsdPhysics.DriveAPI, "angular")
  Add -> Physics -> Joint State (angular) (PhysxSchema.JointStateAPI, "angular")
so that ``set_gains(save_to_usd=True)`` has somewhere to write.

----------------------------------------------------------------------
How to choose the robot/policy (Script Editor cannot pass CLI args):
----------------------------------------------------------------------
From a terminal, persist your selection once::

    python setup_robot_gains.py <policy_dir> <robot_prim>
    python setup_robot_gains.py /path/to/policy/go2_locomotion /go2

The robot_prim is normalized to ``/World/<name>`` (so ``/go2`` and ``go2``
both become ``/World/go2``). The choice is written to
``~/.isaaclegs_setup_gains.json`` and stays there until you run::

    python setup_robot_gains.py clear

In Isaac Sim, open this file in the Script Editor and click Run. ``main()``
reads the sidecar and applies the gains. If no sidecar exists, it falls back
to the built-in G1 defaults.
"""

# ============================================================
# Built-in defaults (used when no sidecar JSON is present).
# ============================================================

# G1 robot prim path on the stage.
G1_ROBOT_PRIM = "/World/g1"

# G1 policy directory, relative to the IsaacLegs repo root. The repo root is
# auto-detected by walking upward looking for ``src/fullbody_controller/policy``.
G1_POLICY_SUBDIR = "g1_locomotion"

# Apply the angular Drive / JointState schemas before writing gains.
APPLY_PHYSICS_SCHEMAS = True

# ============================================================

import json
import os
import re

import numpy as np
import yaml

SIDECAR_PATH = os.path.expanduser("~/.isaaclegs_setup_gains.json")
_MARKER = ("src", "fullbody_controller", "policy")


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


# ---- config helpers ---------------------------------------------------------


def _normalize_prim(prim: str) -> str:
    """Normalize a user-supplied robot prim path to ``/World/<name>``.

    Examples: ``/go2`` -> ``/World/go2``; ``go2`` -> ``/World/go2``;
    ``/World/go2`` -> unchanged.
    """
    if prim.startswith("/World/"):
        return prim
    return "/World/" + prim.lstrip("/")


def _find_repo_root() -> str:
    """Locate the IsaacLegs checkout by walking upward looking for ``src/fullbody_controller/policy``."""
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass  # __file__ is undefined when the file's contents are pasted into Script Editor
    candidates.append(os.getcwd())

    for start in candidates:
        cur = start
        for _ in range(10):
            if os.path.isdir(os.path.join(cur, *_MARKER)):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    raise RuntimeError(
        "Could not locate the IsaacLegs repo root. Run the CLI once from a terminal "
        "to write the sidecar with an explicit policy_dir:\n"
        "  python setup_robot_gains.py <abs/path/to/policy_dir> <robot_prim>")


def _resolve_config() -> dict:
    """Read the sidecar JSON if present; otherwise fall back to G1 defaults."""
    if os.path.isfile(SIDECAR_PATH):
        try:
            with open(SIDECAR_PATH) as f:
                sidecar = json.load(f)
            return {
                "robot_prim": _normalize_prim(sidecar["robot_prim"]),
                "policy_dir": os.path.abspath(os.path.expanduser(sidecar["policy_dir"])),
                "apply_physics_schemas": bool(sidecar.get("apply_physics_schemas", True)),
            }
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"[setup_robot_gains] WARNING: bad sidecar at {SIDECAR_PATH}: {exc}. "
                  "Falling back to built-in G1 defaults.")

    return {
        "robot_prim": G1_ROBOT_PRIM,
        "policy_dir": os.path.join(_find_repo_root(), *_MARKER, G1_POLICY_SUBDIR),
        "apply_physics_schemas": APPLY_PHYSICS_SCHEMAS,
    }


def _save_sidecar(policy_dir: str, robot_prim: str,
                  apply_physics_schemas: bool = True) -> None:
    """Persist the chosen policy_dir + robot_prim so the next Script Editor Run picks them up."""
    abs_policy_dir = os.path.abspath(os.path.expanduser(policy_dir))
    if not os.path.isdir(abs_policy_dir):
        raise SystemExit(f"policy_dir does not exist: {abs_policy_dir}")
    desc = os.path.join(abs_policy_dir, "IO_descriptors.yaml")
    if not os.path.isfile(desc):
        raise SystemExit(
            f"IO_descriptors.yaml not found in {abs_policy_dir}. "
            "Did you point at the right folder?")
    payload = {
        "policy_dir": abs_policy_dir,
        "robot_prim": _normalize_prim(robot_prim),
        "apply_physics_schemas": bool(apply_physics_schemas),
    }
    with open(SIDECAR_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {SIDECAR_PATH}:")
    print(json.dumps(payload, indent=2))


def _clear_sidecar() -> None:
    """Remove the sidecar, reverting to the built-in G1 defaults."""
    if os.path.isfile(SIDECAR_PATH):
        os.remove(SIDECAR_PATH)
        print(f"Removed {SIDECAR_PATH}")
    else:
        print(f"No sidecar at {SIDECAR_PATH} (nothing to remove).")


# ---- physics / gains --------------------------------------------------------


def _apply_angular_schemas(robot_prim_path: str, default_pos_rad: dict) -> None:
    """Apply angular DriveAPI + JointStateAPI to every revolute joint and seed the init pose.

    This is the scripted equivalent of the Stage panel context-menu items
    'Add -> Physics -> Angular Drive' and 'Add -> Physics -> Joint State (angular)'.
    On the stock G1/Go2 USDs these usually already exist, so this acts as a safety net.

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
    robot_prim: str = None,
    policy_dir: str = None,
    apply_physics_schemas: bool = None,
) -> None:
    """Read the YAMLs and write per-joint gains/efforts/armature onto the robot.

    Any argument left as ``None`` is resolved from the sidecar JSON or the
    built-in G1 defaults.
    """
    # Lazy import so the sidecar helpers work outside Isaac Sim.
    from isaacsim.core.prims import SingleArticulation

    cfg = _resolve_config()
    if robot_prim is None:
        robot_prim = cfg["robot_prim"]
    else:
        robot_prim = _normalize_prim(robot_prim)
    if policy_dir is None:
        policy_dir = cfg["policy_dir"]
    if apply_physics_schemas is None:
        apply_physics_schemas = cfg["apply_physics_schemas"]

    io_desc_path = os.path.join(policy_dir, "IO_descriptors.yaml")
    env_yaml_path = os.path.join(policy_dir, "env.yaml")
    print(f"[setup_robot_gains] robot_prim = {robot_prim}")
    print(f"[setup_robot_gains] policy_dir = {policy_dir}")

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
    # Isaac Lab actuator configs use one of: ``effort_limit_sim`` (PhysX-side, preferred)
    # or ``effort_limit`` (actuator-model side). Either may be a scalar OR a dict of
    # joint-regex -> value. When ``effort_limit_sim`` is None we fall back to
    # ``effort_limit`` (matches PhysX's behavior when no sim override is given).
    with open(env_yaml_path) as f:
        env = yaml.load(f, Loader=_IsaacLabLoader)
    actuators = env["scene"]["robot"]["actuators"]
    eff_by_name = {}
    for group_name, group in actuators.items():
        eff = group.get("effort_limit_sim")
        if eff is None:
            eff = group.get("effort_limit")
        if eff is None:
            raise RuntimeError(
                f"actuator group '{group_name}' has no effort_limit / effort_limit_sim "
                f"in {env_yaml_path}; cannot derive a per-joint effort limit.")
        for pattern in group["joint_names_expr"]:
            for n in yaml_names:
                if not re.fullmatch(pattern, n):
                    continue
                if isinstance(eff, dict):
                    # dict form: keys are joint-name regexes, values are per-joint limits
                    val = next((v for p, v in eff.items() if re.fullmatch(p, n)), None)
                    if val is None:
                        raise RuntimeError(
                            f"joint '{n}' matched group '{group_name}' but no entry in its "
                            f"effort_limit dict {list(eff.keys())} matches it.")
                    eff_by_name[n] = float(val)
                else:
                    eff_by_name[n] = float(eff)

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
    import sys

    args = sys.argv[1:]
    positional = [a for a in args if not a.startswith("-")]
    no_schemas = "--no-physics-schemas" in args

    if positional == ["clear"]:
        _clear_sidecar()
    elif len(positional) == 2:
        # CLI: python setup_robot_gains.py <policy_dir> <robot_prim> [--no-physics-schemas]
        _save_sidecar(
            policy_dir=positional[0],
            robot_prim=positional[1],
            apply_physics_schemas=not no_schemas,
        )
    elif not args:
        # Script Editor "Run": no args -> apply gains using the sidecar / defaults.
        main()
    else:
        raise SystemExit(
            "Usage:\n"
            "  python setup_robot_gains.py <policy_dir> <robot_prim> [--no-physics-schemas]\n"
            "  python setup_robot_gains.py clear\n"
            "  (or open this file in the Isaac Sim Script Editor and click Run)")
