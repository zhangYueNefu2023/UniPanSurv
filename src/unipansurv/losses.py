from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def discrete_time_survival_nll(
    hazard_logits: Tensor,
    interval_index: Tensor,
    event: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Censoring-aware negative log-likelihood for discrete-time survival.

    ``interval_index[p]`` is the interval containing the observed event or
    censoring time. An event contributes survival through earlier intervals and
    the hazard in its event interval. A right-censored observation contributes
    survival through its censoring interval.
    """

    if hazard_logits.ndim != 2:
        raise ValueError("hazard_logits must have shape [patients, intervals]")
    batch, intervals = hazard_logits.shape
    interval_index = interval_index.to(device=hazard_logits.device, dtype=torch.long)
    event = event.to(device=hazard_logits.device, dtype=hazard_logits.dtype)
    if interval_index.shape != (batch,) or event.shape != (batch,):
        raise ValueError("interval_index and event must have shape [patients]")
    if torch.any(interval_index < 0) or torch.any(interval_index >= intervals):
        raise ValueError("interval_index is outside the prediction grid")

    time = torch.arange(intervals, device=hazard_logits.device).unsqueeze(0)
    before = time < interval_index.unsqueeze(1)
    through = time <= interval_index.unsqueeze(1)
    log_hazard = -F.softplus(-hazard_logits)
    log_survival_step = -F.softplus(hazard_logits)

    event_log_likelihood = (
        (log_survival_step * before).sum(dim=1)
        + log_hazard.gather(1, interval_index.unsqueeze(1)).squeeze(1)
    )
    censor_log_likelihood = (log_survival_step * through).sum(dim=1)
    log_likelihood = event * event_log_likelihood + (1.0 - event) * censor_log_likelihood
    per_patient = -log_likelihood

    if sample_weight is None:
        return per_patient.mean()
    sample_weight = sample_weight.to(device=hazard_logits.device, dtype=hazard_logits.dtype)
    if sample_weight.shape != (batch,):
        raise ValueError("sample_weight must have shape [patients]")
    return (sample_weight * per_patient).sum() / sample_weight.sum().clamp_min(1e-8)


def cancer_balance_weights(
    cancer_ids: Tensor,
    training_cancer_counts: Tensor,
    gamma: float,
) -> Tensor:
    """Return normalised per-patient weights N_c**(-gamma).

    Counts must come from the full training partition, not the current batch.
    """

    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    cancer_ids = cancer_ids.to(dtype=torch.long)
    counts = training_cancer_counts.to(device=cancer_ids.device, dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("all supplied training cancer counts must be positive")
    weights = counts[cancer_ids].pow(-gamma)
    return weights / weights.mean().clamp_min(1e-8)

