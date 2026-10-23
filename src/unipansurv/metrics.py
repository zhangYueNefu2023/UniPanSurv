from __future__ import annotations

import torch
from torch import Tensor


def harrell_c_index(observed_time: Tensor, event: Tensor, risk_score: Tensor) -> float:
    """Harrell's C-index with half credit for tied risk scores."""

    time = observed_time.detach().cpu().reshape(-1)
    status = event.detach().cpu().bool().reshape(-1)
    risk = risk_score.detach().cpu().reshape(-1)
    concordant = 0.0
    comparable = 0
    for first in range(time.numel()):
        if not bool(status[first]):
            continue
        later = time > time[first]
        if not torch.any(later):
            continue
        difference = risk[first] - risk[later]
        concordant += float((difference > 0).sum())
        concordant += 0.5 * float((difference == 0).sum())
        comparable += int(later.sum())
    return float("nan") if comparable == 0 else concordant / comparable
