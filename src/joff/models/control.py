"""System identification and observer-style control models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model

from .base import BaseModel, batch_inputs, batch_targets, regression_loss


class ARX(BaseModel):
    """Linear autoregressive model with optional exogenous regressors."""

    model_type = "arx"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.input_dim is None or config.output_dim is None:
            raise BuildError(
                "ARX requires input_dim and output_dim. Legal input describes one time-step "
                "feature width and target feature width."
            )
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.ar_order = config.ar_order
        self.exogenous_dim = config.exogenous_dim
        self.regressor_dim = self.input_dim * self.ar_order + self.exogenous_dim
        self.linear = nn.Linear(self.regressor_dim, self.output_dim)

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Predict targets from flattened ARX regressors."""

        regressor = self._regressor(batch)
        prediction = self.linear(regressor)
        return {"prediction": prediction, "regressor": regressor}

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute supervised ARX regression loss."""

        loss = regression_loss(output["prediction"], batch_targets(batch), self.config.loss)
        return {"loss": loss, "losses": {self.config.loss: loss}}

    def _regressor(self, batch: Any) -> torch.Tensor:
        history = _flatten_history(batch_inputs(batch), self.input_dim, self.ar_order)
        exogenous = _exogenous(batch, self.exogenous_dim, history)
        regressor = history if exogenous is None else torch.cat([history, exogenous], dim=1)
        if regressor.shape[1] != self.regressor_dim:
            raise BuildError(
                f"ARX regressor width must be {self.regressor_dim}. "
                f"Current shape: {tuple(regressor.shape)}."
            )
        return regressor


class Observer(BaseModel):
    """Simple learned state observer with prediction and correction steps."""

    model_type = "observer"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.input_dim is None or config.output_dim is None:
            raise BuildError(
                "Observer requires input_dim and output_dim. Legal input describes observed or "
                "control feature width and target feature width."
            )
        state_dim = config.observer_state_dim or config.hidden_size or config.output_dim
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.state_dim = state_dim
        self.sequence_output = config.sequence_output
        self.state_transition = nn.Linear(state_dim, state_dim, bias=False)
        self.input_matrix = nn.Linear(config.input_dim, state_dim, bias=False)
        self.output_matrix = nn.Linear(state_dim, config.output_dim)
        self.observer_gain = nn.Linear(config.output_dim, state_dim, bias=False)

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run observer dynamics over a single step or sequence."""

        x = _as_sequence(batch_inputs(batch), self.input_dim)
        state = x.new_zeros(x.shape[0], self.state_dim)
        outputs: list[torch.Tensor] = []
        for step in range(x.shape[1]):
            state, prediction = self._step(state, x[:, step, :])
            outputs.append(prediction)
        sequence = torch.stack(outputs, dim=1)
        prediction = sequence if self.sequence_output == "all" else sequence[:, -1, :]
        return {"prediction": prediction, "sequence": sequence, "state": state}

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute supervised observer prediction loss."""

        target = _align_target(batch_targets(batch), output["prediction"])
        loss = regression_loss(output["prediction"], target, self.config.loss)
        return {"loss": loss, "losses": {self.config.loss: loss}}

    def _step(self, state: torch.Tensor, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior_state = self.state_transition(state) + self.input_matrix(observation)
        prior_output = self.output_matrix(prior_state)
        measurement = _measurement_from_observation(observation, self.output_dim)
        corrected_state = prior_state + self.observer_gain(measurement - prior_output)
        prediction = self.output_matrix(corrected_state)
        return corrected_state, prediction


def _flatten_history(x: torch.Tensor, input_dim: int, ar_order: int) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[1] != ar_order or x.shape[2] != input_dim:
            raise BuildError(
                f"ARX sequence input must have shape (batch, {ar_order}, {input_dim}). "
                f"Current shape: {tuple(x.shape)}."
            )
        return x.reshape(x.shape[0], ar_order * input_dim)
    if x.ndim == 2:
        expected = ar_order * input_dim
        if x.shape[1] != expected:
            raise BuildError(
                f"ARX flat input width must be {expected}. Current shape: {tuple(x.shape)}."
            )
        return x
    raise BuildError(f"ARX input must be 2D or 3D. Current shape: {tuple(x.shape)}.")


def _exogenous(
    batch: Any,
    exogenous_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if exogenous_dim == 0:
        return None
    if not isinstance(batch, dict):
        raise BuildError(
            f"ARX exogenous_dim={exogenous_dim} requires a dict batch with key 'u' or "
            "'exogenous'."
        )
    for key in ("u", "exogenous", "control", "controls"):
        if key in batch:
            value = batch[key]
            if not isinstance(value, torch.Tensor):
                raise BuildError(f"ARX exogenous key {key!r} must contain a torch.Tensor.")
            if value.ndim != 2 or value.shape != (reference.shape[0], exogenous_dim):
                raise BuildError(
                    f"ARX exogenous tensor must have shape ({reference.shape[0]}, "
                    f"{exogenous_dim}). Current shape: {tuple(value.shape)}."
                )
            return value
    raise BuildError("ARX dict batch must include one of: u, exogenous, control, controls.")


def _as_sequence(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[-1] != input_dim:
            raise BuildError(
                f"Observer input last dimension must match input_dim={input_dim}. "
                f"Current shape: {tuple(x.shape)}."
            )
        return x
    if x.ndim == 2:
        if x.shape[-1] != input_dim:
            raise BuildError(
                f"Observer flat input width must match input_dim={input_dim}. "
                f"Current shape: {tuple(x.shape)}."
            )
        return x.unsqueeze(1)
    raise BuildError(f"Observer input must be 2D or 3D. Current shape: {tuple(x.shape)}.")


def _measurement_from_observation(observation: torch.Tensor, output_dim: int) -> torch.Tensor:
    if observation.shape[1] < output_dim:
        raise BuildError(
            f"Observer input_dim must be at least output_dim for correction. "
            f"Current input width: {observation.shape[1]}, output_dim={output_dim}."
        )
    return observation[:, :output_dim]


def _align_target(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if target.shape == prediction.shape:
        return target
    if prediction.ndim == 2 and target.ndim == 3 and target.shape[0] == prediction.shape[0]:
        return target[:, -1, :]
    if prediction.ndim == 3 and target.ndim == 2 and target.shape[0] == prediction.shape[0]:
        return target.unsqueeze(1).expand(-1, prediction.shape[1], -1)
    return target


register_model("arx", ARX, aliases=("autoregressive_exogenous",), replace=True)
register_model("observer", Observer, aliases=("ndo", "neural_dynamic_observer"), replace=True)
