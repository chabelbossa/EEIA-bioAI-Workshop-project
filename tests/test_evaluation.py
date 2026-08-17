import pandas as pd

from bioai.evaluation import evaluate_checkpoint_on_csv
from bioai.model import StudentMLP, save_checkpoint


def test_evaluation_reports_unseen_surface(tmp_path) -> None:
    train = tmp_path / "train.csv"
    evaluation = tmp_path / "val.csv"
    checkpoint = tmp_path / "student.pt"

    pd.DataFrame(
        {
            "sequence": ["ATG" * 66 + "AT", "CCC" * 66 + "CC"],
            "label": [1, 0],
        }
    ).to_csv(train, index=False)
    pd.DataFrame(
        {
            "sequence": ["ATG" * 66 + "AT", "ACG" * 66 + "AC"],
            "label": [1, 0],
        }
    ).to_csv(evaluation, index=False)
    save_checkpoint(checkpoint, StudentMLP(), threshold=0.5)

    report = evaluate_checkpoint_on_csv(
        checkpoint=checkpoint,
        train_csv=train,
        evaluation_csv=evaluation,
        device="cpu",
    )
    assert report.total_rows == 2
    assert report.canonically_unseen_rows == 1
    assert report.canonically_unseen is not None
    assert report.overlap_audit["evaluation_rows_canonical_overlap"] == 1
