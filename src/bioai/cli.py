"""Command-line interface for feature extraction, audits and inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__
from .audit import audit_csv_files
from .features import FEATURE_DIM, FEATURE_VERSION, advanced_features


def _json_dump(payload: object, output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioai",
        description="Utilities for the EEIA bacterial coding-region project.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    about = subparsers.add_parser("about", help="show the public feature contract")
    about.set_defaults(handler=_handle_about)

    features = subparsers.add_parser("features", help="extract the 761-D DNA vector")
    features.add_argument("--sequence", required=True)
    features.add_argument("--output", help="optional .npy output path")
    features.set_defaults(handler=_handle_features)

    audit = subparsers.add_parser(
        "audit",
        help="audit exact and reverse-complement overlap between CSV splits",
    )
    audit.add_argument("--train", required=True, help="training CSV")
    audit.add_argument("--evaluation", required=True, help="validation/test CSV")
    audit.add_argument("--sequence-column", default="sequence")
    audit.add_argument("--output", help="optional JSON report path")
    audit.set_defaults(handler=_handle_audit)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a checkpoint on all rows and canonically-unseen rows",
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--train", required=True, help="training CSV for overlap audit")
    evaluate.add_argument("--evaluation", required=True, help="validation/test CSV")
    evaluate.add_argument("--sequence-column", default="sequence")
    evaluate.add_argument("--label-column", default="label")
    evaluate.add_argument("--threshold", type=float)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", help="optional JSON report path")
    evaluate.set_defaults(handler=_handle_evaluate)

    predict = subparsers.add_parser("predict", help="predict with a versioned student")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--sequence", required=True)
    predict.add_argument("--threshold", type=float)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(handler=_handle_predict)
    return parser


def _handle_about(_args: argparse.Namespace) -> int:
    _json_dump(
        {
            "package_version": __version__,
            "feature_version": FEATURE_VERSION,
            "feature_dimension": FEATURE_DIM,
            "task": "coding-vs-non-coding DNA window classification",
        },
        None,
    )
    return 0


def _handle_features(args: argparse.Namespace) -> int:
    vector = advanced_features(args.sequence)
    if args.output:
        np.save(args.output, vector)
        print(f"saved {vector.shape[0]} features to {args.output}")
    else:
        _json_dump(
            {
                "feature_version": FEATURE_VERSION,
                "shape": list(vector.shape),
                "dtype": str(vector.dtype),
                "values": vector.tolist(),
            },
            None,
        )
    return 0


def _handle_audit(args: argparse.Namespace) -> int:
    report = audit_csv_files(
        args.train,
        args.evaluation,
        sequence_column=args.sequence_column,
    )
    _json_dump(report.to_dict(), args.output)
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_checkpoint_on_csv

    report = evaluate_checkpoint_on_csv(
        checkpoint=args.checkpoint,
        train_csv=args.train,
        evaluation_csv=args.evaluation,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        threshold=args.threshold,
        device=args.device,
    )
    _json_dump(report.to_dict(), args.output)
    return 0


def _handle_predict(args: argparse.Namespace) -> int:
    from .model import load_checkpoint, predict_probabilities

    model, checkpoint_threshold, metadata = load_checkpoint(args.checkpoint)
    threshold = checkpoint_threshold if args.threshold is None else args.threshold
    vector = advanced_features(args.sequence)[None, :]
    probability = float(
        predict_probabilities(model, vector, device=args.device).reshape(-1)[0]
    )
    _json_dump(
        {
            "probability_coding": probability,
            "prediction": int(probability >= threshold),
            "threshold": threshold,
            "feature_version": FEATURE_VERSION,
            "checkpoint_metadata": metadata,
        },
        None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
