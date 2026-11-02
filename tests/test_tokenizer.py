from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from unipansurv import UniPanSurv, UniPanSurvConfig
from unipansurv.layers import MolecularTokenizer


def config(**overrides) -> UniPanSurvConfig:
    return UniPanSurvConfig(num_features=10, num_source_cancers=2, **overrides)


def test_token_components_preserve_their_meaning() -> None:
    """Changing one input must not alter the other two component slices."""

    torch.manual_seed(24)
    tokenizer = MolecularTokenizer(config()).eval()
    ids = torch.tensor([0, 1, 0, 0])
    assays = torch.tensor([0, 0, 1, 0])
    values = torch.tensor([0.2, 0.2, 0.2, -1.7])
    tokens = tokenizer(ids, assays, values)
    assert tokens.shape == (4, 64)
    # A different identity changes only the first 24 dimensions.
    assert not torch.equal(tokens[0, :24], tokens[1, :24])
    assert torch.equal(tokens[0, 24:], tokens[1, 24:])
    # A different assay changes only the next eight dimensions.
    assert torch.equal(tokens[0, :24], tokens[2, :24])
    assert not torch.equal(tokens[0, 24:32], tokens[2, 24:32])
    assert torch.equal(tokens[0, 32:], tokens[2, 32:])
    # A different measurement changes only the final 32 dimensions.
    assert torch.equal(tokens[0, :32], tokens[3, :32])
    assert not torch.equal(tokens[0, 32:], tokens[3, 32:])
    # Value normalisation is confined to those final 32 dimensions.
    assert torch.allclose(tokens[:, 32:].mean(-1), torch.zeros(4), atol=1e-6)
    assert torch.allclose(tokens[:, 32:].var(-1, unbiased=False), torch.ones(4), atol=1e-3)


def test_value_network_dimensions_and_gradient_paths() -> None:
    tokenizer = MolecularTokenizer(config(dropout=0.0))
    assert tokenizer.feature_embedding.weight.shape == (10, 24)
    assert tokenizer.omics_embedding.weight.shape == (3, 8)
    assert [type(layer) for layer in tokenizer.value_encoder.network] == [
        nn.Linear, nn.GELU, nn.Linear, nn.LayerNorm
    ]
    assert tokenizer.value_encoder.network[0].weight.shape == (32, 1)
    assert tokenizer.value_encoder.network[2].weight.shape == (32, 32)
    values = torch.tensor([0.2, -1.7], requires_grad=True)
    output = tokenizer(torch.tensor([0, 1]), torch.tensor([0, 1]), values)
    (output * torch.arange(1, 65)).sum().backward()
    assert values.grad is not None and torch.all(values.grad != 0)
    for parameter in tokenizer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("override", [
    {"feature_embedding_dim": 0},
    {"omics_embedding_dim": -1},
    {"value_embedding_dim": 64},
])
def test_invalid_component_dimensions_are_rejected(override) -> None:
    with pytest.raises(ValueError, match="embedding"):
        config(**override)


def test_new_configuration_and_weights_round_trip(tmp_path) -> None:
    first = UniPanSurv(config())
    path = tmp_path / "concat.pt"
    torch.save({"config": first.config.to_dict(), "model": first.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    assert [saved["config"][name] for name in (
        "feature_embedding_dim", "omics_embedding_dim", "value_embedding_dim"
    )] == [24, 8, 32]
    second = UniPanSurv(UniPanSurvConfig.from_dict(saved["config"]))
    second.load_state_dict(saved["model"], strict=True)
    assert all(torch.equal(value, second.state_dict()[key])
               for key, value in first.state_dict().items())


@pytest.mark.parametrize("strict", [True, False])
def test_legacy_additive_embeddings_cannot_be_silently_loaded(strict) -> None:
    model = UniPanSurv(config())
    legacy = copy.deepcopy(model.state_dict())
    legacy["tokenizer.feature_embedding.weight"] = torch.zeros(10, 64)
    legacy["tokenizer.omics_embedding.weight"] = torch.zeros(3, 64)
    with pytest.raises(RuntimeError, match="size mismatch for tokenizer"):
        model.load_state_dict(legacy, strict=strict)
