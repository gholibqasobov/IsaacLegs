#!/usr/bin/env python3
"""Launch the Isaac Sim GUI with one of the local robot ``.usd`` scenes in this
folder, optionally dropped into one of Isaac Sim's stock environments.

The script auto-discovers every ``*.usd`` file sitting next to it and exposes each
one as its own command-line 

    conda activate env_isaaclab

    python launch_scene.py --list                          # list scenes + environments
    python launch_scene.py --go2_digital_twin              # robot only, GUI paused
    python launch_scene.py --go2_digital_twin --flat_grid  # robot on the grid floor
    python launch_scene.py --go2_digital_twin --stairs --play   # + start simulating
    python launch_scene.py --g1_flat_scene --headless      # no GUI window

"""

import argparse
import os
import sys
from pathlib import Path

SCENE_DIR = Path(__file__).resolve().parent

# Stock Isaac Sim environments, referenced live from the asset server or add from local device
ENVIRONMENTS = {
    "flat_grid": "/Isaac/Environments/Grid/default_environment.usd",
    "flat_plane": "/Isaac/Environments/Terrains/flat_plane.usd",
    "rough_plane": "/Isaac/Environments/Terrains/rough_plane.usd",
    "simple_room": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "full_warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
}


def discover_scenes():
    """Return a sorted ``{stem: absolute_path}`` map of the ``.usd`` files here."""
    scenes = {}
    for usd in sorted(SCENE_DIR.glob("*.usd")):
        scenes[usd.stem] = str(usd)
    return scenes


def build_parser(scenes):
    parser = argparse.ArgumentParser(
        prog="launch_scene.py",
        description="Open a local .usd scene in the Isaac Sim GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_argument_group("robot scenes (choose one)")
    for stem in scenes:
        group.add_argument(
            f"--{stem}",
            dest="scene",
            action="store_const",
            const=stem,
            help=f"Open {scenes[stem]}",
        )
    env_group = parser.add_argument_group("environment (optional, choose at most one)")
    for name, rel_path in ENVIRONMENTS.items():
        env_group.add_argument(
            f"--{name}",
            dest="env",
            action="store_const",
            const=name,
            help=f"Reference {rel_path}",
        )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available scenes and exit (does not boot Isaac Sim).",
    )
    parser.add_argument(
        "--ros2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the isaacsim.ros2.bridge extension so the .usd action-graph "
        "ROS2 topics are published (default: enabled; use --no-ros2 to disable).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Start the timeline immediately after loading (default: stay paused).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a GUI window.",
    )
    return parser


def print_scenes(scenes):
    if not scenes:
        print(f"No .usd scenes found in {SCENE_DIR}")
        return
    print(f"Available robot scenes in {SCENE_DIR}:")
    width = max(len(s) for s in scenes)
    for stem, path in scenes.items():
        print(f"  --{stem.ljust(width)}  ->  {path}")


def print_environments():
    print("Available environments (referenced from the Isaac asset server):")
    width = max(len(s) for s in ENVIRONMENTS)
    for name, rel_path in ENVIRONMENTS.items():
        print(f"  --{name.ljust(width)}  ->  {rel_path}")


def main():
    scenes = discover_scenes()
    parser = build_parser(scenes)
    args = parser.parse_args()

    if args.list:
        print_scenes(scenes)
        print()
        print_environments()
        return 0

    if not scenes:
        print(f"No .usd scenes found in {SCENE_DIR}", file=sys.stderr)
        return 1

    if args.scene is None:
        print("Error: no scene selected.\n", file=sys.stderr)
        print_scenes(scenes)
        return 1

    usd_path = scenes[args.scene]
    if not os.path.isfile(usd_path):
        print(f"Error: scene file no longer exists: {usd_path}", file=sys.stderr)
        return 1

    # ---- Boot Isaac Sim (Kit modules only exist after SimulationApp is created) ----
    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError:
        print(
            "Error: could not import 'isaacsim'. Activate the Isaac Sim conda env "
            "first, e.g.  conda activate env_isaaclab",
            file=sys.stderr,
        )
        return 1

    config = {
        "width": 1280,
        "height": 720,
        "sync_loads": True,
        "headless": args.headless,
        "renderer": "RaytracedLighting",
    }
    print(f"Booting Isaac Sim and loading: {usd_path}")
    kit = SimulationApp(launch_config=config)

    import carb
    import omni
    from isaacsim.core.utils.stage import is_stage_loading

    # Enable the ROS2 bridge
    if args.ros2:
        from isaacsim.core.utils.extensions import enable_extension

        print("Enabling ROS2 bridge (isaacsim.ros2.bridge)...")
        enable_extension("isaacsim.ros2.bridge")
        kit.update()

    omni.usd.get_context().open_stage(usd_path)

    # Let the stage start loading, then wait for it to finish.
    kit.update()
    kit.update()
    print("Loading stage...")
    while is_stage_loading():
        kit.update()
    print("Loading complete.")

    # Optionally drop the robot into a stock Isaac environment, referenced (not
    # copied) straight from the asset server.
    if args.env:
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        try:
            assets_root = get_assets_root_path()
        except RuntimeError:
            assets_root = None

        if assets_root is None:
            carb.log_error(
                "Could not reach the Isaac asset server; skipping environment "
                f"'{args.env}'. Opening the robot-only scene."
            )
        else:
            env_url = assets_root + ENVIRONMENTS[args.env]
            print(f"Adding environment '{args.env}': {env_url}")
            add_reference_to_stage(env_url, f"/World/{args.env}")
            kit.update()
            kit.update()
            while is_stage_loading():
                kit.update()
            print("Environment loaded.")

    timeline = omni.timeline.get_timeline_interface()
    if args.play:
        print("Starting timeline (--play).")
        timeline.play()
    else:
        print("Scene loaded, timeline paused. Press Play in the GUI to simulate.")
        if args.ros2:
            print(
                "Note: ROS2 topics from the action graphs only publish while the "
                "timeline is playing -- press Play (or re-run with --play)."
            )

    try:
        while kit.is_running():
            kit.update()
    finally:
        timeline.stop()
        kit.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
