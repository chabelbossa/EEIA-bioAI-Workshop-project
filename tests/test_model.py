from pathlib import Path

import numpy as np
import pytest
import torch

from bioai.features import FEATURE_DIM
from bioai.model import (
    StudentMLP,
    load_checkpoint,
    predict_probabilities,
    save_checkpoint,
)


def test_student_forward_shape_and_parameter_count() -> None:
    model = StudentMLP()
    output = model(torch.zeros(4, FEATURE_DIM))
    assert output.shape == (4,)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    assert trainable == 48_961


def test_student_rejects_wrong_feature_dimension() -> None:
    model = StudentMLP()
    with pytest.raises(ValueError, match="expected inputs"):
        model(torch.zeros(2, FEATURE_DIM - 1))


def test_versioned_checkpoint_round_trip(tmp_path: Path) -> None:
    model = StudentMLP()
    path = tmp_path / "student.pt"
    save_checkpoint(path, model, threshold=0.265, metadata={"seed": 42})
    restored, threshold, metadata = load_checkpoint(path)
    assert threshold == pytest.approx(0.265)
    assert metadata == {"seed": 42}
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)


def test_predict_probabilities_range() -> None:
    model = StudentMLP()
    features = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    probabilities = predict_probabilities(model, features, device="cpu")
    assert probabilities.shape == (3,)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_config_and_checkpoint_validation(tmp_path: Path) -> None:
    from bioai.model import StudentConfig, resolve_device

    with pytest.raises(ValueError):
        StudentConfig(input_dim=0)
    with pytest.raises(ValueError):
        StudentConfig(dropout=1.0)
    assert resolve_device("cpu").type == "cpu"

    model = StudentMLP()
    with pytest.raises(ValueError, match="threshold"):
        save_checkpoint(tmp_path / "bad.pt", model, threshold=2.0)

    raw = tmp_path / "raw.pt"
    torch.save(model.state_dict(), raw)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        load_checkpoint(raw)
