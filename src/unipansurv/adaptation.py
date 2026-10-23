from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import UniPanSurvConfig
from .layers import PatientAdapter, RelationLoRA


class TargetAdapter(nn.Module):
    """The only trainable state used for a new cancer in few-shot mode."""

    def __init__(
        self,
        config: UniPanSurvConfig,
        initial_context: Tensor,
        initial_calibration: Tensor,
    ) -> None:
        super().__init__()
        self.context = nn.Parameter(initial_context.detach().clone())
        self.risk_calibration = nn.Parameter(initial_calibration.detach().clone())
        self.relation_lora = RelationLoRA(config.hidden_dim, config.adapter_rank)
        self.patient_adapter = PatientAdapter(config.hidden_dim, config.adapter_bottleneck)

    def reset_from_unknown(self, context: Tensor, calibration: Tensor) -> None:
        """Reset the target-specific context and calibration to Unknown values."""

        with torch.no_grad():
            self.context.copy_(context)
            self.risk_calibration.copy_(calibration)

