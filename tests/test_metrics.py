import pytest

from bioai.metrics import evaluate_binary_probabilities


def test_binary_metrics_are_explicit_and_complete() -> None:
    metrics = evaluate_binary_probabilities(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.6, 0.9],
        threshold=0.5,
    )
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 0
    assert metrics.true_positive == 2
    assert metrics.roc_auc is not None
    assert metrics.average_precision is not None


def test_single_class_ranking_metrics_are_none() -> None:
    metrics = evaluate_binary_probabilities([0, 0], [0.1, 0.2])
    assert metrics.roc_auc is None
    assert metrics.average_precision is None


@pytest.mark.parametrize(
    ("labels", "probabilities", "threshold"),
    [
        ([0, 1], [0.1], 0.5),
        ([0, 2], [0.1, 0.9], 0.5),
        ([0, 1], [-0.1, 0.9], 0.5),
        ([0, 1], [0.1, 0.9], 2.0),
    ],
)
def test_invalid_metric_inputs_are_rejected(labels, probabilities, threshold) -> None:
    with pytest.raises(ValueError):
        evaluate_binary_probabilities(labels, probabilities, threshold=threshold)
