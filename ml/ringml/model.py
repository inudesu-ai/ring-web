"""Model loading and common prediction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .hmm import GaussianHMMClassifier
from .mlp import MLPClassifier


def load_model(path: Path) -> MLPClassifier | GaussianHMMClassifier:
    with np.load(path, allow_pickle=False) as values:
        model_type = str(values["model_type"])
    if model_type == MLPClassifier.model_type:
        return MLPClassifier.load(path)
    if model_type == GaussianHMMClassifier.model_type:
        return GaussianHMMClassifier.load(path)
    raise ValueError(f"Unsupported model type: {model_type}")
