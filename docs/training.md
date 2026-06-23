# How to train a policy

Training a locomotion policy happens in **Isaac Lab**, using NVIDIA's own project template and
RL workflow. This guide does not duplicate that workflow — it is **complementary**, focused on the
one step that matters for deployment: exporting the artifacts IsaacLegs needs (`policy.pt`, `env.yaml` and
`IO_descriptors.yaml`) and getting them into the ROS 2 package, ready to drive a digital twin.

← [Back to the main README](../README.md)

---

## Create and train a new policy

To create a new project and train a policy from scratch, follow NVIDIA's official Isaac Lab documentation:

> **[Isaac Lab — Building your own project / task](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html)**

Once you have a trained policy, return here for the **Export and Deployment** steps below.

Alternatively, you can train one of the projects already in this workspace under
[`isaaclab_projects/`](../isaaclab_projects/):

| Project | Tasks | Policy directory |
|---|---|---|
| `g1_locomotion` | `IsaacLegs-G1-Locomotion-v0`, `IsaacLegs-G1-29DOF-Locomotion-v0` | `g1_locomotion`, `g1_29dof_locomotion` |
| `go2_locomotion` | `IsaacLegs-Go2-Locomotion-v0` | `go2_locomotion` |

Each project ships with its **own auto-generated README** (`isaaclab_projects/<project>/README.md`)
that documents the standard template usage. The walkthrough below follows that same flow with the
G1 humanoid as a concrete example.


---

## Example: train and export the G1 policy

### 1. Install the project (editable)

```bash
cd isaaclab_projects/g1_locomotion
python -m pip install -e source/g1_locomotion
```

### 2. List the available tasks

```bash
python scripts/list_envs.py
```

You should see the IsaacLegs tasks registered:

| Task ID | Robot |
|---|---|
| `IsaacLegs-G1-Locomotion-v0` | G1 |
| `IsaacLegs-G1-29DOF-Locomotion-v0` | G1 (29-DOF) |


### 3. Train

```bash
python scripts/rsl_rl/train.py \
  --task=IsaacLegs-G1-Locomotion-v0 \
  --export_io_descriptors \
  --headless
```

- `--export_io_descriptors` — exports the configuration required by the deployment controller
- Useful flags: `--num_envs`, `--max_iterations`, `--seed`, `--video`.

<!-- MEDIA: docs/assets/train_tensorboard.png — reward curves in TensorBoard (optional). -->

Outputs land under:

```
logs/rsl_rl/<experiment>/<timestamp>/
  io_descriptors/IO_descriptors.yaml
  params/env.yaml
  params/agent.yaml
```

### 4. Export the runnable policy

```bash
python scripts/rsl_rl/play.py --task=IsaacLegs-G1-Locomotion-v0
```

This writes:

```
logs/rsl_rl/<experiment>/<timestamp>/exported/
  policy.pt      # TorchScript — used by the ROS 2 controller
  policy.onnx
```


### 5. Copy the artifacts into the ROS 2 package

The controller loads its policy from `src/fullbody_controller/policy/<project>/`. Copy **all three**
artifacts so deployment is fully self-describing:

```bash
DEST=src/fullbody_controller/policy/g1_locomotion
cp logs/rsl_rl/<experiment>/<timestamp>/exported/policy.pt                  $DEST/
cp logs/rsl_rl/<experiment>/<timestamp>/io_descriptors/IO_descriptors.yaml  $DEST/
cp logs/rsl_rl/<experiment>/<timestamp>/params/env.yaml                     $DEST/
```

> Custom policies can be deployed in the same way. Copy `policy.pt`, `IO_descriptors.yaml`, and `env.yaml` into a directory under src/fullbody_controller/policy/. The controller loads the policy and configuration directly from these files.


## Next →

[**Create a digital twin of your robot**](digital_twin.md)
