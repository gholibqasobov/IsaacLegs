# Using the policy controller for a custom robot

The [`fullbody_controller`](../src/fullbody_controller) node runs **any** Isaac Lab locomotion
policy on **any** legged robot, with **no per-robot code**. This guide explains the contract that
makes that possible, how to extend the controller for new sensors/observations, and how to launch
it for your robot.

← [Back to creating Robot Digital Twin](digital_twin.md)

---

## Prerequisite

A working digital twin that publishes standard ROS 2 topics — at minimum
`joint_states` (`sensor_msgs/JointState`) and `imu` (`sensor_msgs/Imu`) — and accepts
`joint_command` (`sensor_msgs/JointState` position targets). See
[the digital twin guide](digital_twin.md).

---

## How it works: the `IO_descriptors.yaml` contract

At startup the controller loads `IO_descriptors.yaml` (next to `policy.pt`, or via the
`io_descriptors_path` parameter) and configures itself entirely from it:

- **Actions** — one *or more* joint-position terms, each with its own joint set, default offset,
  and scale. The controller flattens them in descriptor order (the layout the policy network
  emits) and applies `target = default + action × scale` per joint. A whole-body policy can, for
  example, split `legs` (scale 0.5) and `upper_body` (scale 0.25).
- **Observations** — the `observations.policy` term list, in order. Each joint-based term carries
  its own `joint_names`, so the **observation joint order is decoupled from the action joint
  order** — the controller selects joints by name at runtime.

Because everything is read from the descriptor, swapping robots or retraining is a matter of
swapping YAML + `policy.pt` — no code changes.

> The controller forces `use_sim_time = True` and runs at 50 Hz, driven by a `TimeSynchronizer`
> over `joint_states` + `imu`.

---

## Extending the controller: three registries

Everything the node can subscribe to, produce, and publish lives in three class-level registries
in [`policy_controller.py`](../src/fullbody_controller/fullbody_controller/policy_controller.py).
Extending is a registry edit, not a rewrite.

### `OBS_PRODUCERS` — add an observation term

Maps an IO-descriptor term name → a method returning that term's 1-D slice. To add a term:

```python
# 1. add the mapping
OBS_PRODUCERS = {
    # ...
    "my_term": "_obs_my_term",
}

# 2. implement the producer (ctx carries the per-tick inputs; term is the descriptor entry)
def _obs_my_term(self, ctx, term):
    return some_1d_numpy_array
```

The dispatch loop walks the descriptor's terms in order, calls each producer, applies any
`scale`/`clip` overloads generically, and concatenates. **An unknown term name fails fast at
startup** with the list of known terms.

### `TOPIC_SOURCES` — add a sensor/input

Declarative subscriptions. Two modes:

- `"tick"` — synchronized inputs that drive the control loop (today: `joint_states`, then `imu`).
- `"async"` — cached by a callback, read by producers. Can be **gated** by `feeds_terms` so the
  subscription is skipped unless the loaded descriptor actually uses it.

The file ships a **commented LiDAR template** — the canonical worked example for adding a sensor:

```python
# {"key": "lidar",  "param": "lidar_topic",  "default": "",
#  "msg_type": LaserScan, "mode": "async", "callback": "_lidar_callback",
#  "qos": "default", "feeds_terms": ("lidar_scan",)},
```

To add it: import the message type, add the `TOPIC_SOURCES` entry, seed a cache attribute in
`_init_topic_caches`, add the `_lidar_callback`, and add the matching `OBS_PRODUCERS` producer.

> Adding a third **tick** source requires updating `_tick`'s signature (its positional args bind
> to the `tick`-mode entries in registry order).

### `PUBLISHERS` — add an output

```python
PUBLISHERS = [
    {"key": "joint_command", "param": "joint_command_topic", "default": "joint_command",
     "msg_type": JointState, "attr": "_joint_publisher", "qos": "sim"},
]
```

---

## Create a launch file

Per-robot launch files are thin wrappers around `policy_controller.launch.py` that just point at
the right policy. Copy
[`g1_29dof.launch.py`](../src/fullbody_controller/launch/g1_29dof.launch.py):

```python
def generate_launch_description():
    share = get_package_share_directory('fullbody_controller')
    base = os.path.join(share, 'launch', 'policy_controller.launch.py')
    policy_path = os.path.join(share, 'policy', 'my_robot_locomotion', 'policy.pt')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base),
            launch_arguments={'policy_path': policy_path}.items(),
        ),
    ])
```

---

## Run it

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch fullbody_controller my_robot.launch.py
```

Key parameters (`-p name:=value`, or as launch args):

| Parameter | Default | Meaning |
|---|---|---|
| `policy_path` | shipped checkpoint | path to the TorchScript policy (`.pt`) |
| `io_descriptors_path` | next to the policy | override the descriptor location |
| `decimation` | `1` | run the policy every Nth tick |
| `warmup_sec` | `0.0` | ease into the default pose before engaging the policy |
| `warmup_interpolate` | `True` | interpolate from spawn pose to default (no snap) |
| `*_topic` | per registry | remap any subscribed/published topic (e.g. `joint_command_topic`) |

> **Warmup** matters for a clean start: it drives the robot to the policy's default pose so the
> first observation is in-distribution, avoiding a violent snap or an out-of-distribution input.

---

← [Back to the main README](../README.md)
