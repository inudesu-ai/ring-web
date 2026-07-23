#!/usr/bin/env python3
"""Train one left-to-right Gaussian HMM per gesture using EM."""

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
from ringml.hmm import GaussianHMMClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Ring Sound HMM model.")
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
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument(
        "--topology", choices=("left_right", "ergodic"), default="ergodic"
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=1000,
        help="Balanced EM cap per class; zero uses all windows.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def balanced_cap(dataset, maximum: int, seed: int):
    if maximum <= 0:
        return dataset
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(dataset.labels.tolist())):
        indices = np.flatnonzero(dataset.labels == label)
        rng.shuffle(indices)
        selected.extend(indices[:maximum].tolist())
    return dataset.subset(np.asarray(selected, dtype=np.int64))


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
    hmm_train = balanced_cap(train, args.max_windows_per_class, args.seed)
    classes = np.asarray(sorted(set(dataset.labels.tolist())), dtype=str)
    model = GaussianHMMClassifier(
        classes=classes,
        target_steps=args.target_steps,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        state_count=args.states,
        topology=args.topology,
    )
    histories = model.fit(
        hmm_train.windows,
        hmm_train.labels,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
    )
    train_probabilities = model.predict_proba(hmm_train.windows)
    validation_probabilities = model.predict_proba(validation.windows)
    test_probabilities = model.predict_proba(test.windows)
    train_predicted = model.classes[np.argmax(train_probabilities, axis=1)]
    validation_predicted = model.classes[
        np.argmax(validation_probabilities, axis=1)
    ]
    test_predicted = model.classes[np.argmax(test_probabilities, axis=1)]
    confidence = np.max(validation_probabilities, axis=1)
    correct = validation_predicted == validation.labels
    recommended_threshold = 0.7
    for threshold in np.linspace(0.5, 0.95, 10):
        accepted = confidence >= threshold
        if np.any(accepted) and np.mean(correct[accepted]) >= 0.9:
            recommended_threshold = float(threshold)
            break

    scores = {
        "train": classification_scores(
            hmm_train.labels, train_predicted, model.classes
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
            "training_windows": len(hmm_train.windows),
            "validation_windows": len(validation.windows),
            "test_windows": len(test.windows),
            "recommended_threshold": recommended_threshold,
            "split_method": split_method,
            "scores": scores,
            "fit_diagnosis": diagnosis,
            "em_iterations": {
                label: len(history) for label, history in histories.items()
            },
            "seed": args.seed,
        }
    )
    model.save(args.output)

    print(
        "TRAIN (BALANCED EM SUBSET)\n"
        + format_metrics(hmm_train.labels, train_predicted, model.classes)
    )
    print(
        "\nVALIDATION\n"
        + format_metrics(
            validation.labels, validation_predicted, model.classes
        )
    )
    print("\nTEST\n" + format_metrics(test.labels, test_predicted, model.classes))
    print(
        f"\nrecommended threshold: {recommended_threshold:.2f} | "
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
