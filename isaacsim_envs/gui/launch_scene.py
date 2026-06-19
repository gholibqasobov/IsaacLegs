#!/usr/bin/env python3
"""Launch the Isaac Sim GUI with one of the local ``.usd`` scenes in this folder.

The script auto-discovers every ``*.usd`` file sitting next to it and exposes each
one as its own command-line flag (named after the file stem). Pick exactly one.

Usage (run with the ``env_isaaclab`` conda env active so ``isaacsim`` is importable)::

    conda activate env_isaaclab

    python launch_scene.py --list                 # show available scenes, then exit
    python launch_scene.py --go2_digital_twin      # open the scene, GUI paused
    python launch_scene.py --go2_digital_twin --play   # open and start simulating
    python launch_scene.py --g1_flat_scene --headless  # no GUI window

The scenes are local files, so they are opened by absolute path via
``omni.usd.get_context().open_stage(...)`` (no Omniverse asset server needed).
"""

import argparse
import os
import sys
from pathlib import Path

SCENE_DIR = Path(__file__).resolve().parent


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
    group = parser.add_argument_group("scenes (choose one)")
    for stem in scenes:
        group.add_argument(
            f"--{stem}",
            dest="scene",
            action="store_const",
            const=stem,
            help=f"Open {scenes[stem]}",
        )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available scenes and exit (does not boot Isaac Sim).",
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
    print(f"Available scenes in {SCENE_DIR}:")
    width = max(len(s) for s in scenes)
    for stem, path in scenes.items():
        print(f"  --{stem.ljust(width)}  ->  {path}")


def main():
    scenes = discover_scenes()
    parser = build_parser(scenes)
    args = parser.parse_args()

    if args.list:
        print_scenes(scenes)
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

    omni.usd.get_context().open_stage(usd_path)

    # Let the stage start loading, then wait for it to finish.
    kit.update()
    kit.update()
    print("Loading stage...")
    while is_stage_loading():
        kit.update()
    print("Loading complete.")

    timeline = omni.timeline.get_timeline_interface()
    if args.play:
        print("Starting timeline (--play).")
        timeline.play()
    else:
        print("Scene loaded, timeline paused. Press Play in the GUI to simulate.")

    try:
        while kit.is_running():
            kit.update()
    finally:
        timeline.stop()
        kit.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
