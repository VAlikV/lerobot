# Frozen DINOv2-S + ACT

This opt-in experiment shares one frozen `facebook/dinov2-small` encoder between
`front` and `gripper`. Patch features retain their 16×16 spatial grid at 224×224;
the 384→512 projection and ACT are trained. Existing ResNet defaults are unchanged.

From this checkout, install the optional dependency into your training environment:

```bash
pip install -e '.[transformers-dep]'
python kuka/act/train_dinov2.py --dataset-root ../kuka_put_cap \
  --output-dir ../outputs/kuka_put_cap/act_dinov2 --device cuda
```

Defaults: batch 16, 70,000 updates, seed 42, chunk 60, absolute actions, LR 1e-5,
weight decay 1e-4, gradient clipping 10. Both 224×224 camera videos must be present.
All episode video paths are checked before downloading weights. Use `--help` for
CLI options. Dataset/output defaults are relative to the checkout's parent,
independent of the shell working directory; pass explicit paths in other layouts.
This is a fresh-training script, not an optimizer-resume or validation pipeline.
Use a separate empty output directory for every run.

Images must be RGB float tensors in [0, 1]. The saved ACT preprocessor applies
ImageNet mean/std exactly once for DINOv2. Do not run an additional Hugging Face
image processor. State/action normalization still uses dataset statistics.
The backbone stays in eval mode with no gradients even during `policy.train()`.

Initial training downloads DINOv2 weights. The complete backbone architecture
and weights are subsequently saved inside each policy checkpoint. Restore a
local checkpoint offline with its saved processors:

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

path = '/absolute/path/to/checkpoint'
policy = ACTPolicy.from_pretrained(path, local_files_only=True, strict=True)
pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
policy.reset()
action = post(policy.select_action(pre(observation)))
```

The Transformers dependency is still required for offline restoration. The
checkpoint's configured device must be available (or load its config and change
`device` before passing it to `from_pretrained`). Each episode requires `reset()`.
Default execution queries the policy every step without temporal ensembling.

Compare against ResNet using the same episodes, batch size and update budget.
Measure model latency and the full camera/control loop separately before 30 Hz
execution. No robot control is launched by this training script.
