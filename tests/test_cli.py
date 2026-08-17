import json

from bioai.cli import main
from bioai.features import FEATURE_DIM


def test_about_command(capsys) -> None:
    assert main(["about"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["feature_dimension"] == FEATURE_DIM


def test_features_command_writes_npy(tmp_path) -> None:
    output = tmp_path / "features.npy"
    assert main(
        ["features", "--sequence", "ATG" * 66 + "AT", "--output", str(output)]
    ) == 0
    assert output.exists()


def test_audit_command_writes_report(tmp_path) -> None:
    import pandas as pd

    train = tmp_path / "train.csv"
    evaluation = tmp_path / "evaluation.csv"
    output = tmp_path / "audit.json"
    pd.DataFrame({"sequence": ["ATGCAA"]}).to_csv(train, index=False)
    pd.DataFrame({"sequence": ["TTGCAT"]}).to_csv(evaluation, index=False)
    assert main(
        [
            "audit",
            "--train",
            str(train),
            "--evaluation",
            str(evaluation),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text())
    assert payload["evaluation_rows_canonical_overlap"] == 1


def test_predict_command_uses_versioned_checkpoint(tmp_path, capsys) -> None:
    from bioai.model import StudentMLP, save_checkpoint

    checkpoint = tmp_path / "student.pt"
    save_checkpoint(checkpoint, StudentMLP(), threshold=0.265, metadata={"seed": 42})
    assert main(
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--sequence",
            "ATG" * 66 + "AT",
            "--device",
            "cpu",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["threshold"] == 0.265
    assert payload["checkpoint_metadata"] == {"seed": 42}
