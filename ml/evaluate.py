#!/usr/bin/env python3
"""Evaluate a saved gesture model against labeled JSONL sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ringml.data import expand_paths, format_metrics, load_sessions, window_sessions
from ringml.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Ring Sound model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    sessions = load_sessions(expand_paths(args.data))
    dataset = window_sessions(
        sessions,
        window_seconds=model.window_seconds,
        stride_seconds=model.stride_seconds,
        target_steps=model.target_steps,
    )
    probabilities = model.predict_proba(dataset.windows)
    predicted = model.classes[np.argmax(probabilities, axis=1)]
    print(format_metrics(dataset.labels, predicted, model.classes))


if __name__ == "__main__":
    main()
