"""Minimal feature-extraction example; no embedding or checkpoint required."""

from bioai.features import FEATURE_VERSION, advanced_features

sequence = "ATG" * 66 + "AT"
features = advanced_features(sequence)
print(FEATURE_VERSION, features.shape, features.dtype)
