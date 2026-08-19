import pytest
import torch

from bioai.distillation import binary_distillation_loss


def test_alpha_one_equals_hard_bce() -> None:
    student = torch.tensor([0.2, -0.4], requires_grad=True)
    teacher = torch.tensor([1.2, -1.0])
    labels = torch.tensor([1.0, 0.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(student, labels)
    actual = binary_distillation_loss(
        student,
        teacher,
        labels,
        alpha=1.0,
        temperature=2.0,
    )
    assert torch.allclose(actual, expected)


def test_distillation_loss_has_finite_gradients() -> None:
    student = torch.tensor([0.2, -0.4], requires_grad=True)
    teacher = torch.tensor([1.2, -1.0])
    labels = torch.tensor([1.0, 0.0])
    loss = binary_distillation_loss(student, teacher, labels, alpha=0.5, temperature=2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize(
    ("alpha", "temperature"),
    [(-0.1, 1.0), (1.1, 1.0), (0.5, 0.0), (0.5, -1.0)],
)
def test_invalid_parameters_are_rejected(alpha: float, temperature: float) -> None:
    logits = torch.zeros(2)
    labels = torch.zeros(2)
    with pytest.raises(ValueError):
        binary_distillation_loss(
            logits,
            logits,
            labels,
            alpha=alpha,
            temperature=temperature,
        )


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        binary_distillation_loss(
            torch.zeros(2),
            torch.zeros(3),
            torch.zeros(2),
        )


def test_label_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="labels"):
        binary_distillation_loss(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(3),
        )


def test_seed_everything_is_repeatable() -> None:
    from bioai.distillation import seed_everything

    seed_everything(7)
    first = torch.rand(3)
    seed_everything(7)
    second = torch.rand(3)
    assert torch.equal(first, second)
