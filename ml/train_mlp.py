#!/usr/bin/env python3
"""Train a compact multi-class neural network on Ring Sound IMU windows."""

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
from ringml.mlp import MLPClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Ring Sound MLP model.")
    parser.add_argument("--data", nargs="+", required=True, help="JSONL paths or globs.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz model.")
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--stride-seconds", type=float, default=0.4)
    parser.add_argument("--target-steps", type=int, default=40)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def recommend_threshold(
    expected: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    for threshold in np.linspace(0.5, 0.95, 10):
        accepted = confidence >= threshold
        if np.any(accepted) and np.mean(predicted[accepted] == expected[accepted]) >= 0.9:
            return float(threshold)
    return 0.7


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
    model = MLPClassifier(
        classes=classes,
        target_steps=args.target_steps,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    history = model.fit(
        train.windows,
        train.labels,
        validation.windows,
        validation.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        patience=args.patience,
    )
    probabilities = model.predict_proba(validation.windows)
    predicted = model.classes[np.argmax(probabilities, axis=1)]
    threshold = recommend_threshold(validation.labels, probabilities, model.classes)
    model.metadata.update(
        {
            "training_sessions": len(set(train.groups.tolist())),
            "validation_sessions": len(set(validation.groups.tolist())),
            "training_windows": len(train.windows),
            "validation_windows": len(validation.windows),
            "recommended_threshold": threshold,
            "seed": args.seed,
        }
    )
    model.save(args.output)

    print(format_metrics(validation.labels, predicted, model.classes))
    print(
        f"\nepochs: {len(history)} | "
        f"recommended threshold: {threshold:.2f} | model: {args.output}"
    )


if __name__ == "__main__":
    main()
