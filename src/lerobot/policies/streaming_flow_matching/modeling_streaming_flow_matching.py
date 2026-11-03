"""Streaming Flow Policy: advance an action-space flow one physical tick at a time."""

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_streaming_flow_matching import StreamingFlowMatchingConfig
from .flow import sample_conditional_flow, sample_flow_time
from .network import ObservationEncoder, StreamingVelocityNet
from .processor_streaming_flow_matching import ANCHOR, VALID_INTERVALS


@dataclass
class FlowState:
    sample: Tensor
    conditioning: Tensor
    steps: int = 0


class StreamingFlowMatchingPolicy(PreTrainedPolicy):
    config_class = StreamingFlowMatchingConfig
    name = "streaming_flow_matching"

    def __init__(self, config: StreamingFlowMatchingConfig, dataset_meta: Any | None = None, **kwargs):
        super().__init__(config)
        if dataset_meta is not None:
            config.validate_dataset_metadata(dataset_meta)
        config.validate_features()
        self.observation_encoder = ObservationEncoder(config)
        self.velocity_net = StreamingVelocityNet(config, self.observation_encoder.output_dim)
        self.reset()

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._observation_history: dict[str, deque] = {}
        self._flow_state: FlowState | None = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        trajectory = batch[ACTION]
        if trajectory.ndim != 3 or trajectory.shape[1] != self.config.horizon:
            raise ValueError("SFP training requires a preprocessed measured trajectory")
        valid = batch[VALID_INTERVALS]
        # Singleton episode tails cannot supply a derivative. Drop these rows rather
        # than inventing zero velocities or discarding all useful short windows.
        rows = valid.any(-1)
        if not rows.any():
            raise ValueError("Batch has no valid measured trajectory intervals")
        trajectory, valid = trajectory[rows], valid[rows]
        observations = {key: batch[key][rows] for key in (OBS_STATE, *self.config.image_features)}
        time = sample_flow_time(valid)
        sample, target = sample_conditional_flow(
            trajectory,
            time,
            variant=self.config.variant,
            sigma=self.config.sigma,
            sigma1=self.config.sigma1,
            gain=self.config.stabilizing_gain,
        )
        conditioning = self.observation_encoder(observations)
        prediction = self.velocity_net(sample, time, conditioning)
        loss = functional.mse_loss(prediction, target)
        return loss, {"velocity_mse": loss.detach().item(), "valid_samples": rows.sum().item()}

    def _history(self, batch: dict[str, Tensor], *, update: bool) -> dict[str, Tensor]:
        result = {}
        for key in (OBS_STATE, *self.config.image_features):
            value = batch[key]
            single_rank = 2 if key == OBS_STATE else 4
            if value.ndim == single_rank + 1:
                if value.shape[1] != self.config.n_obs_steps:
                    raise ValueError(f"Expected {self.config.n_obs_steps} history frames for {key}")
                frames = list(value.unbind(1))
            elif value.ndim == single_rank:
                previous = list(self._observation_history.get(key, []))
                if previous and previous[-1].shape != value.shape:
                    raise ValueError("Batch size changed during streaming; call reset() first")
                frames = (previous + [value])[-self.config.n_obs_steps :]
                frames = [frames[0]] * (self.config.n_obs_steps - len(frames)) + frames
            else:
                raise ValueError(f"Unexpected inference shape for {key}: {tuple(value.shape)}")
            result[key] = torch.stack(frames, 1)
            if update:
                self._observation_history[key] = deque(
                    (frame.detach().clone() for frame in frames), maxlen=self.config.n_obs_steps
                )
        return result

    def _start_flow(self, history: dict[str, Tensor], anchor: Tensor) -> FlowState:
        action = anchor.clone()
        if self.config.variant == "stochastic":
            sample = torch.stack((action, torch.randn_like(action)), 1)
        else:
            sample = action[:, None]
        return FlowState(sample=sample, conditioning=self.observation_encoder(history))

    def _advance(self, state: FlowState) -> tuple[Tensor, FlowState]:
        substeps = self.config.integration_steps_per_action
        dt = 1 / ((self.config.horizon - 1) * substeps)
        sample = state.sample
        for substep in range(substeps):
            time_value = (state.steps * substeps + substep) * dt
            time = torch.full((sample.shape[0],), time_value, device=sample.device, dtype=torch.float32)
            sample = sample + dt * self.velocity_net(sample, time, state.conditioning)
        next_state = replace(state, sample=sample, steps=state.steps + 1)
        return sample[:, 0].clone(), next_state

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        history = self._history(batch, update=True)
        if self._flow_state is None or self._flow_state.steps >= self.config.n_action_steps:
            self._flow_state = self._start_flow(history, batch[ANCHOR])
        action, self._flow_state = self._advance(self._flow_state)
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Preview the next execution horizon without changing streaming session state."""
        # A preview must not alter the latent sampled at the next real refresh.
        devices = [batch[ANCHOR].device] if batch[ANCHOR].is_cuda else []
        with torch.random.fork_rng(devices=devices):
            history = self._history(batch, update=False)
            state = self._flow_state
            if state is not None:
                state = replace(state, sample=state.sample.clone(), conditioning=state.conditioning.clone())
            actions = []
            for _ in range(self.config.n_action_steps):
                if state is None or state.steps >= self.config.n_action_steps:
                    state = self._start_flow(history, batch[ANCHOR])
                action, state = self._advance(state)
                actions.append(action)
            return torch.stack(actions, 1)
