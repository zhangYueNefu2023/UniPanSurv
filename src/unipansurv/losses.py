from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import SurvivalHead


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


def categorical_survival_nll(
    logits: Tensor,
    interval_index: Tensor,
    event: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Likelihood for event-time categories plus a final beyond-grid category."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [patients, intervals + 1]")
    batch, categories = logits.shape
    intervals = categories - 1
    interval_index = interval_index.to(device=logits.device, dtype=torch.long)
    event = event.to(device=logits.device, dtype=logits.dtype)
    if interval_index.shape != (batch,) or event.shape != (batch,):
        raise ValueError("interval_index and event must have shape [patients]")
    if torch.any(interval_index < 0) or torch.any(interval_index >= intervals):
        raise ValueError("interval_index is outside the prediction grid")

    log_probability = F.log_softmax(logits, dim=-1)
    losses: list[Tensor] = []
    for index in range(batch):
        observed_interval = int(interval_index[index].item())
        if bool(event[index] > 0.5):
            log_likelihood = log_probability[index, observed_interval]
        else:
            log_likelihood = torch.logsumexp(
                log_probability[index, observed_interval + 1 :], dim=0
            )
        losses.append(-log_likelihood)
    per_patient = torch.stack(losses)
    if sample_weight is None:
        return per_patient.mean()
    weight = sample_weight.to(device=logits.device, dtype=logits.dtype)
    return (weight * per_patient).sum() / weight.sum().clamp_min(1e-8)


def deephit_ranking_loss(
    logits: Tensor,
    interval_index: Tensor,
    event: Tensor,
    temperature: float = 0.10,
) -> Tensor:
    """Single-event DeepHit-style pairwise ranking term."""

    probability = torch.softmax(logits, dim=-1)[:, :-1]
    cumulative_risk = torch.cumsum(probability, dim=-1)
    interval_index = interval_index.to(device=logits.device, dtype=torch.long)
    event = event.to(device=logits.device, dtype=torch.bool)
    terms: list[Tensor] = []
    for first in range(logits.size(0)):
        if not bool(event[first]):
            continue
        comparable = interval_index > interval_index[first]
        if not torch.any(comparable):
            continue
        horizon = interval_index[first]
        margin = (
            cumulative_risk[first, horizon]
            - cumulative_risk[comparable, horizon]
        )
        terms.append(torch.exp(-margin / temperature).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def cox_partial_likelihood_loss(
    log_risk: Tensor,
    observed_time: Tensor,
    event: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Negative Breslow partial log-likelihood for one mini-batch risk set.

    The training CLI labels this as a mini-batch Cox ablation.  For an exact
    cohort-wide Cox objective, use a batch containing the complete risk set.
    """

    score = log_risk.reshape(-1)
    time = observed_time.to(device=score.device, dtype=score.dtype).reshape(-1)
    event = event.to(device=score.device, dtype=torch.bool).reshape(-1)
    if score.numel() != time.numel() or score.numel() != event.numel():
        raise ValueError("Cox inputs must have the same length")
    if not torch.any(event):
        return score.sum() * 0.0
    weight = (
        torch.ones_like(score)
        if sample_weight is None
        else sample_weight.to(device=score.device, dtype=score.dtype).reshape(-1)
    )
    contributions: list[Tensor] = []
    contribution_weights: list[Tensor] = []
    for value in torch.unique(time[event]):
        at_time = event & (time == value)
        risk_set = time >= value
        denominator = torch.logsumexp(score[risk_set], dim=0)
        contributions.append(score[at_time] - denominator)
        contribution_weights.append(weight[at_time])
    contribution = torch.cat(contributions)
    event_weight = torch.cat(contribution_weights)
    return -(event_weight * contribution).sum() / event_weight.sum().clamp_min(1e-8)


def survival_objective(
    *,
    head: SurvivalHead,
    objective_logits: Tensor,
    hazard_logits: Tensor,
    interval_index: Tensor,
    event: Tensor,
    observed_time: Tensor,
    sample_weight: Tensor | None = None,
    deephit_ranking_weight: float = 0.20,
    deephit_ranking_temperature: float = 0.10,
) -> Tensor:
    """Dispatch the loss used by each survival-head ablation."""

    if head == "discrete":
        return discrete_time_survival_nll(
            hazard_logits, interval_index, event, sample_weight
        )
    if head == "mtlr":
        return categorical_survival_nll(
            objective_logits, interval_index, event, sample_weight
        )
    if head == "deephit":
        likelihood = categorical_survival_nll(
            objective_logits, interval_index, event, sample_weight
        )
        ranking = deephit_ranking_loss(
            objective_logits,
            interval_index,
            event,
            temperature=deephit_ranking_temperature,
        )
        return likelihood + deephit_ranking_weight * ranking
    if head == "cox":
        return cox_partial_likelihood_loss(
            objective_logits, observed_time, event, sample_weight
        )
    raise ValueError(f"unsupported survival head: {head}")

