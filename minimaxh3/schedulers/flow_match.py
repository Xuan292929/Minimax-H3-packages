# SPDX-License-Identifier: Apache-2.0
"""
Native PyTorch Flow Matching and Euler Ancestral Scheduler for MiniMax-H3.
Zero dependency on sglang or diffusers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch


class MiniMaxH3FlowMatchScheduler:
    """
    Flow Matching Scheduler with time-shift for MiniMax-H3 audio-video generation and training.
    """

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.sigmas: torch.Tensor = torch.tensor([])
        self.timesteps: torch.Tensor = torch.tensor([])
        self.training: bool = False
        self.linear_timesteps_weights: torch.Tensor = torch.tensor([])

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        shift: float = 2.22,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.training = training
        base = torch.linspace(denoising_strength, 0.0, num_inference_steps + 1, dtype=torch.float32)[:-1]
        sigmas = shift * base / (1.0 + (shift - 1.0) * base)
        timesteps = sigmas * self.num_train_timesteps
        self.sigmas = sigmas
        self.timesteps = timesteps

        if training:
            self._set_training_weights()

        return self.sigmas, self.timesteps

    def _set_training_weights(self) -> None:
        steps = float(self.num_train_timesteps)
        x = self.sigmas * steps
        y = torch.exp(-2.0 * (((x - steps / 2.0) / steps) ** 2))
        y_shifted = y - y.min()
        sum_val = y_shifted.sum()
        if sum_val > 0:
            bsmntw_weighing = y_shifted * (steps / sum_val)
        else:
            bsmntw_weighing = torch.ones_like(y_shifted)
        if len(self.timesteps) != int(steps) and len(self.timesteps) > 1:
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.linear_timesteps_weights.numel() == 0:
            return torch.tensor(1.0, device=timestep.device)
        timestep_id = torch.argmin((self.timesteps.to(timestep.device) - timestep.float()).abs())
        return self.linear_timesteps_weights[timestep_id].to(timestep.device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        sample: torch.Tensor,
        to_final: bool = False,
    ) -> torch.Tensor:
        """
        Advance one step along the flow matching trajectory.
        prev_sample = sample + model_output * (sigma_next - sigma_curr)
        Note: model_output is -v (negated velocity), so:
        sample - v * (sigma_next - sigma_curr) = sample + v * (sigma_curr - sigma_next).
        """
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.cpu()
        else:
            t_val = torch.tensor(timestep)

        timestep_id = torch.argmin((self.timesteps - t_val).abs()).item()
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 0.0
        else:
            sigma_next = self.sigmas[timestep_id + 1].item()

        dt = float(sigma_next - sigma)
        prev_sample = sample + model_output * dt
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """
        Flow matching interpolation:
        x_t = (1 - sigma) * x_0 + sigma * eps
        """
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.cpu()
        else:
            t_val = torch.tensor(timestep)
        timestep_id = torch.argmin((self.timesteps - t_val).abs()).item()
        sigma = float(self.sigmas[timestep_id])
        sample = (1.0 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: Union[torch.Tensor, float] = None,
    ) -> torch.Tensor:
        """
        Flow matching target:
        v = noise - sample (x_1 - x_0)
        """
        return noise - sample


__all__ = ["MiniMaxH3FlowMatchScheduler"]
