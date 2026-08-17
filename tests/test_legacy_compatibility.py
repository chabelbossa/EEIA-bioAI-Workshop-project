"""Guard the public feature package against silent drift from the notebook code."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bioai.features import advanced_features


LEGACY_PATH = Path("day3/src/advanced_features.py")


@pytest.mark.skipif(not LEGACY_PATH.exists(), reason="legacy workshop module not available")
def test_feature_vector_matches_legacy_implementation() -> None:
    spec = importlib.util.spec_from_file_location("legacy_features", LEGACY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sequence = "ATGCGTAC" * 25
    current = advanced_features(sequence)
    legacy = module.advanced_feature_matrix([sequence])[0]
    np.testing.assert_allclose(current, legacy, rtol=0.0, atol=0.0)
