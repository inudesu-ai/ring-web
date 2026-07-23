#!/usr/bin/env python3
"""Train one left-to-right Gaussian HMM per gesture using EM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ringml.data import (
    expand_paths,
    format_metrics,
    grouped_train_validation_split,
    load_sessions,
    window_sessions,
)
from ringml.hmm import GaussianHMMClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Ring Sound HMM model.")
    parser.add_argument("--data", nargs="+", required=True, help="JSONL paths or globs.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz model.")
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--stride-seconds", type=float, default=0.4)
    parser.add_argument("--target-steps", type=int, default=40)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = expand_paths(args.data)
    sessions = load_sessions(paths)
    dataset = window_sessions(
        sessions,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        target_steps=args.target_steps,
    )
    train, validation = grouped_train_validation_split(
        dataset,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    classes = np.asarray(sorted(set(dataset.labels.tolist())), dtype=str)
    model = GaussianHMMClassifier(
        classes=classes,
        target_steps=args.target_steps,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        state_count=args.states,
    )
    histories = model.fit(
        train.windows,
        train.labels,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
    )
    probabilities = model.predict_proba(validation.windows)
    predicted = model.classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    correct = predicted == validation.labels
    recommended_threshold = 0.7
    for threshold in np.linspace(0.5, 0.95, 10):
        accepted = confidence >= threshold
        if np.any(accepted) and np.mean(correct[accepted]) >= 0.9:
            recommended_threshold = float(threshold)
            break

    model.metadata.update(
        {
            "training_sessions": len(set(train.groups.tolist())),
            "validation_sessions": len(set(validation.groups.tolist())),
            "training_windows": len(train.windows),
            "validation_windows": len(validation.windows),
            "recommended_threshold": recommended_threshold,
            "em_iterations": {
                label: len(history) for label, history in histories.items()
            },
            "seed": args.seed,
        }
    )
    model.save(args.output)

    print(format_metrics(validation.labels, predicted, model.classes))
    print(
        f"\nrecommended threshold: {recommended_threshold:.2f} | "
        f"model: {args.output}"
    )


if __name__ == "__main__":
    main()
