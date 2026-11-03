"""Torch equivalents of the conditional flows in siddancha/streaming-flow-policy.

Reference: ea08aa9bf24fe9960abc85d5ed69023ce7a91cc8, core/sfp_aspace.py
and pusht/{sfpd,sfps}.py. Derivatives are with respect to normalized flow time,
not seconds. No simulator, spline library, or ODE dependency is required.
"""

import torch
from torch import Tensor


def interpolate_trajectory(trajectory: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate first-order hold and its derivative for (B, H, D) waypoints."""
    intervals = trajectory.shape[1] - 1
    position = time.clamp(0, 1) * intervals
    left = position.long().clamp(max=intervals - 1)
    row = torch.arange(trajectory.shape[0], device=trajectory.device)
    start, end = trajectory[row, left], trajectory[row, left + 1]
    value = start + (position - left).unsqueeze(-1) * (end - start)
    return value, (end - start) * intervals


def sample_flow_time(valid_intervals: Tensor) -> Tensor:
    """Uniformly sample valid intervals, then a uniform position inside each one."""
    if not valid_intervals.any(dim=-1).all():
        raise ValueError("Each training sample needs at least one valid trajectory interval")
    left = torch.multinomial(valid_intervals.float(), 1).squeeze(-1)
    fraction = torch.rand(left.shape, device=left.device)
    return (left + fraction) / valid_intervals.shape[-1]


def sample_conditional_flow(
    trajectory: Tensor,
    time: Tensor,
    *,
    variant: str,
    sigma: float,
    sigma1: float = 0.1,
    gain: float = 0.0,
    noise: Tensor | None = None,
    latent: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return network input and target velocities, shaped (B, 1 or 2, D)."""
    mean, derivative = interpolate_trajectory(trajectory, time)
    noise = torch.randn_like(mean) if noise is None else noise
    time = time[:, None]
    if variant == "deterministic":
        error = sigma * torch.exp(-gain * time) * noise
        return (mean + error)[:, None], (derivative - gain * error)[:, None]
    if variant != "stochastic":
        raise ValueError(f"Unknown flow variant: {variant}")
    latent = torch.randn_like(mean) if latent is None else latent
    sigma_r = (sigma1**2 - sigma**2) ** 0.5
    action = mean + sigma * noise + sigma_r * time * latent
    z = (1 - (1 - sigma1) * time) * latent + time * mean
    va = derivative + sigma_r * latent
    vz = mean + time * derivative - (1 - sigma1) * latent
    return torch.stack((action, z), 1), torch.stack((va, vz), 1)
