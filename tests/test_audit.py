import numpy as np

from bioai.audit import audit_sequence_overlap, canonically_unseen_mask
from bioai.features import reverse_complement


def test_overlap_audit_counts_exact_and_reverse_complement_rows() -> None:
    train = ["ATGCAA", "CCCCAA", "TTTTGG"]
    evaluation = [
        "ATGCAA",
        reverse_complement("CCCCAA"),
        "ACGTAC",
    ]
    report = audit_sequence_overlap(train, evaluation)
    assert report.exact_overlap_unique == 1
    assert report.canonical_overlap_unique == 2
    assert report.evaluation_rows_exact_overlap == 1
    assert report.evaluation_rows_canonical_overlap == 2
    assert report.evaluation_rows_canonically_unseen == 1


def test_unseen_mask_preserves_evaluation_order() -> None:
    mask = canonically_unseen_mask(["ATGCAA"], ["TTGCAT", "ACGTAC"])
    assert mask.dtype == np.bool_
    assert mask.tolist() == [False, True]


def test_audit_csv_files(tmp_path) -> None:
    import pandas as pd

    from bioai.audit import audit_csv_files

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    pd.DataFrame({"sequence": ["ATGCAA"]}).to_csv(train_path, index=False)
    pd.DataFrame({"sequence": ["TTGCAT", "ACGTAC"]}).to_csv(val_path, index=False)
    report = audit_csv_files(train_path, val_path)
    assert report.evaluation_rows_canonical_overlap == 1
