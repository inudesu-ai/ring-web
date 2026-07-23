#!/usr/bin/env python3
"""Train, tune, and evaluate HMM/MLP models on the million-row simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.data import (  # noqa: E402
    classification_scores,
    diagnose_fit,
    expand_paths,
    grouped_train_validation_test_split,
    load_sessions,
    window_sessions,
)
from ringml.hmm import GaussianHMMClassifier  # noqa: E402
from ringml.mlp import MLPClassifier  # noqa: E402


def evaluate(model, dataset) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(dataset.windows)
    predicted = model.classes[np.argmax(probabilities, axis=1)]
    return (
        classification_scores(dataset.labels, predicted, model.classes),
        predicted,
        probabilities,
    )


def balanced_cap(dataset, maximum: int, seed: int):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(dataset.labels.tolist())):
        indices = np.flatnonzero(dataset.labels == label)
        rng.shuffle(indices)
        selected.extend(indices[:maximum].tolist())
    return dataset.subset(np.asarray(selected, dtype=np.int64))


def confusion(expected: np.ndarray, predicted: np.ndarray, classes: np.ndarray):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    index = {label: value for value, label in enumerate(classes)}
    for truth, guess in zip(expected, predicted):
        matrix[index[truth], index[guess]] += 1
    return matrix


def fit_mlp_candidates(train, validation, classes, args):
    candidates = [
        {
            "name": "compact-48",
            "hidden_size": 48,
            "dropout": 0.24,
            "l2": 8e-4,
            "noise_std": 0.010,
            "learning_rate": 1.2e-3,
        },
        {
            "name": "regularized-64",
            "hidden_size": 64,
            "dropout": 0.18,
            "l2": 3e-4,
            "noise_std": 0.008,
            "learning_rate": 1e-3,
        },
        {
            "name": "capacity-96",
            "hidden_size": 96,
            "dropout": 0.12,
            "l2": 1e-4,
            "noise_std": 0.006,
            "learning_rate": 8e-4,
        },
    ]
    runs = []
    for candidate_index, config in enumerate(candidates):
        print(f"\nTraining MLP candidate: {config['name']}", flush=True)
        model = MLPClassifier(
            classes=classes,
            target_steps=args.target_steps,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            hidden_size=config["hidden_size"],
            seed=args.seed + candidate_index,
        )
        history = model.fit(
            train.windows,
            train.labels,
            validation.windows,
            validation.labels,
            epochs=args.mlp_epochs,
            batch_size=args.batch_size,
            learning_rate=config["learning_rate"],
            l2=config["l2"],
            dropout=config["dropout"],
            patience=args.patience,
            noise_std=config["noise_std"],
        )
        train_scores, _, _ = evaluate(model, train)
        validation_scores, _, _ = evaluate(model, validation)
        gap = train_scores["macro_f1"] - validation_scores["macro_f1"]
        print(
            f"{config['name']}: train F1={train_scores['macro_f1']:.4f}, "
            f"validation F1={validation_scores['macro_f1']:.4f}, gap={gap:.4f}",
            flush=True,
        )
        runs.append(
            {
                "config": config,
                "model": model,
                "history": history,
                "train_scores": train_scores,
                "validation_scores": validation_scores,
            }
        )
    def selection_score(run):
        train_f1 = run["train_scores"]["macro_f1"]
        validation_f1 = run["validation_scores"]["macro_f1"]
        gap = max(0.0, train_f1 - validation_f1 - 0.05)
        underfit = max(0.0, 0.80 - train_f1)
        return validation_f1 - 0.7 * gap - 0.5 * underfit

    return max(runs, key=selection_score), runs


def plot_results(output: Path, report: dict, mlp_runs: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in mlp_runs:
        epochs = [int(row["epoch"]) for row in run["history"]]
        axes[0].plot(
            epochs,
            [row["validation_loss"] for row in run["history"]],
            label=run["config"]["name"],
        )
        axes[1].plot(
            epochs,
            [row["validation_accuracy"] for row in run["history"]],
            label=run["config"]["name"],
        )
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_yscale("log")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "mlp_learning_curves.png", dpi=160)
    plt.close(figure)

    classes = report["classes"]
    models = ("mlp", "hmm")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, name in zip(axes, models):
        matrix = np.asarray(report["models"][name]["test_confusion"])
        row_sum = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        normalized = matrix / row_sum
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
        axis.set_title(f"{name.upper()} subject-held-out test")
        axis.set_xticks(range(len(classes)), classes, rotation=55, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output / "test_confusion_matrices.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1M-row gesture experiment.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--stride-seconds", type=float, default=0.8)
    parser.add_argument("--target-steps", type=int, default=32)
    parser.add_argument("--mlp-epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hmm-states", type=int, default=7)
    parser.add_argument("--hmm-iterations", type=int, default=18)
    parser.add_argument("--hmm-train-per-class", type=int, default=700)
    parser.add_argument("--hmm-eval-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    started = time.monotonic()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("records") != 1_000_000:
        raise ValueError("This experiment requires exactly 1,000,000 records")
    paths = expand_paths(args.data)
    print(f"Loading {len(paths)} compressed shards...", flush=True)
    sessions = load_sessions(paths)
    print(f"Loaded {len(sessions)} sessions; creating windows...", flush=True)
    dataset = window_sessions(
        sessions,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        target_steps=args.target_steps,
    )
    train, validation, test, split_method = grouped_train_validation_test_split(
        dataset, use_predefined=True
    )
    classes = np.asarray(sorted(set(dataset.labels.tolist())), dtype=str)
    subject_sets = {
        "train": set(train.subjects.tolist()),
        "validation": set(validation.subjects.tolist()),
        "test": set(test.subjects.tolist()),
    }
    if (
        subject_sets["train"] & subject_sets["validation"]
        or subject_sets["train"] & subject_sets["test"]
        or subject_sets["validation"] & subject_sets["test"]
    ):
        raise RuntimeError("Subject leakage detected")

    chosen_mlp, mlp_runs = fit_mlp_candidates(
        train, validation, classes, args
    )
    mlp = chosen_mlp["model"]
    mlp_train_scores = chosen_mlp["train_scores"]
    mlp_validation_scores = chosen_mlp["validation_scores"]
    mlp_test_scores, mlp_test_prediction, mlp_test_probabilities = evaluate(
        mlp, test
    )
    mlp_diagnosis = diagnose_fit(
        mlp_train_scores, mlp_validation_scores, mlp_test_scores
    )
    mlp.metadata.update(
        {
            "dataset_records": manifest["records"],
            "selected_candidate": chosen_mlp["config"],
            "scores": {
                "train": mlp_train_scores,
                "validation": mlp_validation_scores,
                "test": mlp_test_scores,
            },
            "fit_diagnosis": mlp_diagnosis,
            "split_method": split_method,
            "recommended_threshold": 0.65,
        }
    )
    mlp.save(model_dir / "gesture-mlp-sim-1m.npz")

    hmm_train = balanced_cap(train, args.hmm_train_per_class, args.seed)
    hmm_validation = balanced_cap(
        validation, args.hmm_eval_per_class, args.seed + 1
    )
    hmm_test = balanced_cap(test, args.hmm_eval_per_class, args.seed + 2)
    print(
        f"\nTraining HMM on {len(hmm_train.windows):,} balanced windows...",
        flush=True,
    )
    hmm = GaussianHMMClassifier(
        classes=classes,
        target_steps=args.target_steps,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        state_count=args.hmm_states,
        topology="ergodic",
    )
    hmm_histories = hmm.fit(
        hmm_train.windows,
        hmm_train.labels,
        max_iterations=args.hmm_iterations,
    )
    hmm_train_scores, _, _ = evaluate(hmm, hmm_train)
    hmm_validation_scores, _, _ = evaluate(hmm, hmm_validation)
    hmm_test_scores, hmm_test_prediction, _ = evaluate(hmm, hmm_test)
    hmm_diagnosis = diagnose_fit(
        hmm_train_scores, hmm_validation_scores, hmm_test_scores
    )
    hmm.metadata.update(
        {
            "dataset_records": manifest["records"],
            "scores": {
                "train": hmm_train_scores,
                "validation": hmm_validation_scores,
                "test": hmm_test_scores,
            },
            "fit_diagnosis": hmm_diagnosis,
            "balanced_em_windows_per_class": args.hmm_train_per_class,
            "em_iterations": {
                label: len(history) for label, history in hmm_histories.items()
            },
            "split_method": split_method,
            "recommended_threshold": 0.65,
        }
    )
    hmm.save(model_dir / "gesture-hmm-sim-1m.npz")

    report = {
        "schema": "ring-million-experiment/v1",
        "records": manifest["records"],
        "classes": classes.tolist(),
        "sessions": len(sessions),
        "split_method": split_method,
        "subject_leakage": False,
        "subjects": {name: len(values) for name, values in subject_sets.items()},
        "windows": {
            "all": len(dataset.windows),
            "train": len(train.windows),
            "validation": len(validation.windows),
            "test": len(test.windows),
            "hmm_train_balanced": len(hmm_train.windows),
            "hmm_validation_balanced": len(hmm_validation.windows),
            "hmm_test_balanced": len(hmm_test.windows),
        },
        "models": {
            "mlp": {
                "selected_candidate": chosen_mlp["config"],
                "candidate_validation": [
                    {
                        "name": run["config"]["name"],
                        "train": run["train_scores"],
                        "validation": run["validation_scores"],
                        "epochs": len(run["history"]),
                    }
                    for run in mlp_runs
                ],
                "train": mlp_train_scores,
                "validation": mlp_validation_scores,
                "test": mlp_test_scores,
                "fit_diagnosis": mlp_diagnosis,
                "test_mean_confidence": float(
                    np.mean(np.max(mlp_test_probabilities, axis=1))
                ),
                "test_confusion": confusion(
                    test.labels, mlp_test_prediction, classes
                ).tolist(),
            },
            "hmm": {
                "train": hmm_train_scores,
                "validation": hmm_validation_scores,
                "test": hmm_test_scores,
                "fit_diagnosis": hmm_diagnosis,
                "test_confusion": confusion(
                    hmm_test.labels, hmm_test_prediction, classes
                ).tolist(),
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_results(args.output, report, mlp_runs)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
