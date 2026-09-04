# LeRobot Setup Guide

A setup guide for running LeRobot with HIL-SERL (Human-in-the-Loop Sample-Efficient Reinforcement Learning) and the `gym-hil` simulation environment.

## Prerequisites

- Ubuntu 22.04
- `sudo` privileges for installing system packages — if you don't have them (e.g. on a shared server), see [Installation without sudo](#installation-without-sudo)
- Git

## Installation

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh
chmod +x miniconda3.sh
./miniconda3.sh -b
conda config --set auto_activate_base false
```

### 2. Create and Activate the Conda Environment

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

### 3. Install FFmpeg

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

### 4. Install System Dependencies

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev
```

> No `sudo`? This is the only step that needs root. See [Installation without sudo](#installation-without-sudo) for a conda-based replacement.

### 5. Clone the Repositories

```bash
git clone https://github.com/VAlikV/lerobot.git
git clone https://github.com/syedjameel/gym-hil.git
```

### 6. Install LeRobot

```bash
cd lerobot
pip install -e .
pip install -e ".[hilserl]"
```

### 7. Patch `transformers`

`self.post_init()` must be added at the end of the `__init__()` method of the `PreTrainedModel` class in `transformers/modeling_utils.py`. With the `lerobot` environment active, this script applies the patch automatically (it locates the file by importing transformers, and is safe to re-run — no editor or root access needed):

```bash
python - <<'PY'
import ast
from pathlib import Path
import transformers.modeling_utils as m

path = Path(m.__file__)
src = path.read_text()
marker = "self.post_init()  # lerobot-hilserl patch"
if marker in src:
    print(f"already patched: {path}")
else:
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "PreTrainedModel")
    init = next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    lines = src.splitlines(keepends=True)
    lines.insert(init.end_lineno, f"        {marker}\n")
    path.write_text("".join(lines))
    print(f"patched: {path}")
PY
```

To do it by hand instead, open the printed path in any editor (`nano`, `vim`, or `gedit` on a desktop machine), find `class PreTrainedModel`, and add `self.post_init()` as the last line of its `__init__()` method.

Verify:

```bash
grep -n "self.post_init()" $(python -c "import transformers.modeling_utils as m; print(m.__file__)") | head -1
```

### 8. Install Additional Dependencies

```bash
pip install matplotlib
```

## Installation without sudo

Only step 4 (`sudo apt-get install ...`) requires root — it provides the compilers and FFmpeg development headers that `pip` needs when building packages from source. If you don't have root access, get the same toolchain from conda-forge into your user-owned environment instead:

```bash
conda activate lerobot
conda install -y -c conda-forge compilers cmake pkg-config ffmpeg=7.1.1
```

The conda-forge `ffmpeg` package includes the `libav*` development headers, and `pkg-config` from the same environment will find them during `pip install`. Skip step 4 and continue from step 5 as normal.

**Known pitfall — `evdev` fails to compile** with an error like `'KEY_LINK_PHONE' undeclared`:

`evdev` *generates* its C code by scanning the host's kernel headers in `/usr/include/linux/`, but *compiles* it with conda's compiler, which uses conda's own (older) bundled sysroot headers. If the host headers are newer than conda's, the generated code references key codes the conda headers don't have. Fix it by installing the prebuilt evdev from conda-forge before retrying, so pip doesn't need to compile it at all:

```bash
conda install -y -c conda-forge evdev
pip install -e .
```

(Alternative: `conda install -y -c conda-forge 'kernel-headers_linux-64>=6.12'` updates conda's sysroot headers so the compile side catches up with the host.) If another package fails to build for a similar toolchain reason, check whether conda-forge ships it prebuilt and install it the same way.

## Running the Simulation

### 1. Install `gym-hil`

```bash
cd gym-hil
pip install -e .
```

### 2. Launch the Simulation Environment

```bash
cd ../lerobot
python -m lerobot.rl.gym_manipulator --config_path src/lerobot/rl/panda_sim_usb_env.json
```

Use this step to record your offline demonstrations before proceeding to training.

## Training

After recording the offline demonstrations, training requires **two terminals running in parallel**.

### Terminal 1 — Learner

```bash
conda activate lerobot
cd lerobot
python -m lerobot.rl.learner --config_path src/lerobot/rl/panda_sim_usb_train.json
```

### Terminal 2 — Actor

```bash
conda activate lerobot
cd lerobot
python -m lerobot.rl.actor --config_path src/lerobot/rl/panda_sim_usb_train.json
```

## Distributed Training (learner on a server, actor on the robot PC)

The actor and learner communicate over a single gRPC connection. The **learner is the server**: it binds to `policy.actor_learner_config.learner_host:learner_port` from the train config. The **actor is the client**: it dials the same `learner_host:learner_port` from *its* copy of the config. All traffic (transitions with camera frames going up, policy parameters coming down) flows through this one port, and the actor initiates the connection — the server never connects back to your PC.

Since both processes read the same config field but need different values, keep one copy of the train config per machine and change only `actor_learner_config`:

**On the server (learner)** — bind to all interfaces so remote actors can reach it:

```json
"actor_learner_config": {
    "learner_host": "0.0.0.0",
    "learner_port": 50051,
    ...
}
```

**On the robot PC (actor)** — point at the server's IP address:

```json
"actor_learner_config": {
    "learner_host": "192.168.1.42",
    "learner_port": 50051,
    ...
}
```

> Everything else in the two config files (policy settings, dataset repo id, etc.) must stay identical.

Then start the learner first, and the actor after it:

```bash
# on the server
python -m lerobot.rl.learner --config_path <train config>

# on the robot PC
python -m lerobot.rl.actor --config_path <train config>
```

### If the server doesn't accept inbound connections (firewall, shared cluster)

Tunnel the port over SSH instead of opening it. On the robot PC:

```bash
ssh -N -L 50051:127.0.0.1:50051 <user>@<server>
```

Keep that running, leave the server config's `learner_host` as `"127.0.0.1"` (no `0.0.0.0` needed), and set the actor config's `learner_host` to `"127.0.0.1"` too — the tunnel forwards your PC's local port 50051 to the server's. This also encrypts the traffic, which is worth doing anyway on untrusted networks since the gRPC channel itself is insecure (plaintext).

### Notes

- The connection carries camera frames for every transition, so you want a reasonably fast uplink from the robot PC. If the actor logs show the transition queue backing up, network bandwidth is the first suspect.
- Test connectivity before a training run: `nc -zv <server ip> 50051` from the robot PC (or with the SSH tunnel up, `nc -zv 127.0.0.1 50051`).

## Troubleshooting

- **`transformers` patch not applied**: If you encounter initialization errors related to `PreTrainedModel`, double-check that `self.post_init()` was added to the correct `__init__()` method in `modeling_utils.py`.
- **FFmpeg version conflicts**: Ensure that the conda-forge FFmpeg (7.1.1) is being used inside the activated environment, not the system FFmpeg.
- **Path issues**: All `python -m` commands should be run from the root of the `lerobot` repository.

## References

- [LeRobot (fork)](https://github.com/VAlikV/lerobot)
- [gym-hil](https://github.com/syedjameel/gym-hil)