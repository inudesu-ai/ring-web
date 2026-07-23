#!/usr/bin/env python3
"""Train a compact multi-class neural network on Ring Sound IMU windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ringml.data import (
    classification_scores,
    diagnose_fit,
    expand_paths,
    format_metrics,
    grouped_train_validation_test_split,
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
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--group-by", choices=("auto", "subject", "session"), default="auto"
    )
    parser.add_argument("--predefined-split", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path, default=None)
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
    train, validation, test, split_method = grouped_train_validation_test_split(
        dataset,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_by=args.group_by,
        use_predefined=args.predefined_split,
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
        l2=args.l2,
        dropout=args.dropout,
        patience=args.patience,
        noise_std=args.noise_std,
    )
    train_probabilities = model.predict_proba(train.windows)
    validation_probabilities = model.predict_proba(validation.windows)
    test_probabilities = model.predict_proba(test.windows)
    train_predicted = model.classes[np.argmax(train_probabilities, axis=1)]
    validation_predicted = model.classes[
        np.argmax(validation_probabilities, axis=1)
    ]
    test_predicted = model.classes[np.argmax(test_probabilities, axis=1)]
    threshold = recommend_threshold(
        validation.labels, validation_probabilities, model.classes
    )
    scores = {
        "train": classification_scores(
            train.labels, train_predicted, model.classes
        ),
        "validation": classification_scores(
            validation.labels, validation_predicted, model.classes
        ),
        "test": classification_scores(test.labels, test_predicted, model.classes),
    }
    diagnosis = diagnose_fit(
        scores["train"], scores["validation"], scores["test"]
    )
    model.metadata.update(
        {
            "training_sessions": len(set(train.groups.tolist())),
            "validation_sessions": len(set(validation.groups.tolist())),
            "test_sessions": len(set(test.groups.tolist())),
            "training_windows": len(train.windows),
            "validation_windows": len(validation.windows),
            "test_windows": len(test.windows),
            "recommended_threshold": threshold,
            "split_method": split_method,
            "scores": scores,
            "fit_diagnosis": diagnosis,
            "seed": args.seed,
        }
    )
    model.save(args.output)

    print("TRAIN\n" + format_metrics(train.labels, train_predicted, model.classes))
    print(
        "\nVALIDATION\n"
        + format_metrics(
            validation.labels, validation_predicted, model.classes
        )
    )
    print("\nTEST\n" + format_metrics(test.labels, test_predicted, model.classes))
    print(
        f"\nepochs: {len(history)} | "
        f"recommended threshold: {threshold:.2f} | "
        f"fit: {diagnosis['status']} | model: {args.output}"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "model_type": model.model_type,
                    "classes": model.classes.tolist(),
                    "scores": scores,
                    "fit_diagnosis": diagnosis,
                    "metadata": model.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
