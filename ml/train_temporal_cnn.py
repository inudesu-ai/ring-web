#!/usr/bin/env python3
"""Train and export a lightweight temporal CNN for ring gestures."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from ringml.data import (
    classification_scores,
    diagnose_fit,
    expand_paths,
    grouped_train_validation_test_split,
    load_sessions,
    window_sessions,
)
from ringml.temporal_cnn import TemporalCNNClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "PyTorch is required for training. Install ml/requirements-train.txt"
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--stride-seconds", type=float, default=1.6)
    parser.add_argument(
        "--deployment-stride-seconds",
        type=float,
        default=0.4,
        help="Realtime prediction interval stored in the exported model.",
    )
    parser.add_argument("--target-steps", type=int, default=40)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--predefined-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def add_magnitudes(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            values,
            np.linalg.norm(values[:, :, :3], axis=2, keepdims=True),
            np.linalg.norm(values[:, :, 3:], axis=2, keepdims=True),
        ),
        axis=2,
    )


class DepthwiseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_channels,
            input_channels,
            5,
            padding=2 * dilation,
            dilation=dilation,
            groups=input_channels,
        )
        self.pointwise = nn.Conv1d(input_channels, output_channels, 1)
        self.skip = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.relu(self.depthwise(values))
        transformed = self.dropout(self.pointwise(transformed))
        return torch.relu(self.skip(values) + transformed)


class TinyTemporalCNN(nn.Module):
    def __init__(
        self,
        class_count: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        *,
        block_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "feature_mean",
            torch.tensor(feature_mean, dtype=torch.float32).view(1, 1, 8),
        )
        self.register_buffer(
            "feature_std",
            torch.tensor(feature_std, dtype=torch.float32).view(1, 1, 8),
        )
        self.stem = nn.Conv1d(8, 24, 3, padding=1)
        self.block1 = DepthwiseBlock(24, 32, 1, block_dropout)
        self.block2 = DepthwiseBlock(32, 32, 2, block_dropout)
        self.block3 = DepthwiseBlock(32, 32, 4, block_dropout)
        self.attention = nn.Conv1d(32, 1, 1)
        self.fc1 = nn.Linear(96, 48)
        self.head_dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(48, class_count)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        magnitudes = torch.stack(
            (
                torch.linalg.vector_norm(windows[:, :, :3], dim=2),
                torch.linalg.vector_norm(windows[:, :, 3:], dim=2),
            ),
            dim=2,
        )
        features = torch.cat((windows, magnitudes), dim=2)
        values = ((features - self.feature_mean) / self.feature_std).transpose(1, 2)
        values = torch.relu(self.stem(values))
        values = self.block3(self.block2(self.block1(values)))
        attention = torch.softmax(self.attention(values), dim=2)
        pooled = torch.cat(
            (
                torch.sum(values * attention, dim=2),
                torch.mean(values, dim=2),
                torch.amax(values, dim=2),
            ),
            dim=1,
        )
        hidden = self.head_dropout(torch.relu(self.fc1(pooled)))
        return self.fc2(hidden)


def random_rotation(
    windows: torch.Tensor, maximum_degrees: float
) -> torch.Tensor:
    count = len(windows)
    axis = torch.randn(count, 3, device=windows.device)
    axis /= torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
    angle = (
        (torch.rand(count, device=windows.device) * 2.0 - 1.0)
        * np.deg2rad(maximum_degrees)
    )
    cross = torch.zeros(count, 3, 3, device=windows.device)
    cross[:, 0, 1] = -axis[:, 2]
    cross[:, 0, 2] = axis[:, 1]
    cross[:, 1, 0] = axis[:, 2]
    cross[:, 1, 2] = -axis[:, 0]
    cross[:, 2, 0] = -axis[:, 1]
    cross[:, 2, 1] = axis[:, 0]
    identity = torch.eye(3, device=windows.device).expand(count, 3, 3)
    rotation = (
        identity
        + torch.sin(angle)[:, None, None] * cross
        + (1.0 - torch.cos(angle))[:, None, None] * (cross @ cross)
    )
    augmented = windows.clone()
    augmented[:, :, :3] = windows[:, :, :3] @ rotation
    augmented[:, :, 3:] = windows[:, :, 3:] @ rotation
    return augmented


def stress_windows(windows: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic unseen-placement/noise/packet-timing stress set."""

    rng = np.random.default_rng(seed)
    output = np.asarray(windows, dtype=np.float64).copy()
    for index in range(len(output)):
        axis = rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-9)
        angle = rng.uniform(-np.deg2rad(18.0), np.deg2rad(18.0))
        cross = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3)
            + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * (cross @ cross)
        )
        output[index, :, :3] = output[index, :, :3] @ rotation
        output[index, :, 3:] = output[index, :, 3:] @ rotation
        shift = int(rng.integers(-4, 5))
        if shift > 0:
            output[index, shift:] = output[index, :-shift]
            output[index, :shift] = output[index, shift]
        elif shift < 0:
            amount = -shift
            output[index, :-amount] = output[index, amount:]
            output[index, -amount:] = output[index, -amount - 1]
    output *= rng.normal(1.0, 0.025, (len(output), 1, 6))
    bias_scale = np.asarray([0.00065] * 3 + [0.00040] * 3)
    noise_scale = np.asarray([0.00038] * 3 + [0.00009] * 3)
    output += rng.normal(size=(len(output), 1, 6)) * bias_scale
    output += rng.normal(size=output.shape) * noise_scale
    return output


def predictions(
    model: nn.Module,
    windows: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for begin in range(0, len(windows), 1024):
            logits.append(model(windows[begin : begin + 1024].to(device)).cpu())
    values = torch.cat(logits)
    probabilities = torch.softmax(values, dim=1).numpy()
    return np.argmax(probabilities, axis=1), probabilities


def scores(
    labels: np.ndarray,
    predicted_indices: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    return classification_scores(labels, classes[predicted_indices], classes)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    bins: int = 10,
) -> float:
    expected = np.asarray(
        [{label: index for index, label in enumerate(classes)}[label] for label in labels]
    )
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    result = 0.0
    for lower in np.linspace(0.0, 0.9, bins):
        mask = (confidence >= lower) & (confidence < lower + 1.0 / bins)
        if np.any(mask):
            result += np.mean(mask) * abs(
                np.mean(predicted[mask] == expected[mask]) - np.mean(confidence[mask])
            )
    return float(result)


def threshold_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    target_precision: float = 0.99,
) -> dict[str, float]:
    expected = np.asarray(
        [{label: index for index, label in enumerate(classes)}[label] for label in labels]
    )
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    selected = 0.95
    precision = 0.0
    coverage = 0.0
    for threshold in np.linspace(0.50, 0.95, 46):
        accepted = confidence >= threshold
        if np.any(accepted):
            current = float(np.mean(predicted[accepted] == expected[accepted]))
            if current >= target_precision:
                selected = float(threshold)
                precision = current
                coverage = float(np.mean(accepted))
                break
    return {
        "threshold": selected,
        "accepted_precision": precision,
        "coverage": coverage,
    }


def confusion(
    labels: np.ndarray, predicted_indices: np.ndarray, classes: np.ndarray
) -> list[list[int]]:
    class_index = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for label, predicted in zip(labels, predicted_indices):
        matrix[class_index[label], predicted] += 1
    return matrix.tolist()


def export_parameters(model: TinyTemporalCNN) -> dict[str, np.ndarray]:
    state = model.state_dict()
    result = {
        "stem_weight": state["stem.weight"].cpu().numpy(),
        "stem_bias": state["stem.bias"].cpu().numpy(),
        "attention_weight": state["attention.weight"].cpu().numpy(),
        "attention_bias": state["attention.bias"].cpu().numpy(),
        "fc1_weight": state["fc1.weight"].cpu().numpy(),
        "fc1_bias": state["fc1.bias"].cpu().numpy(),
        "fc2_weight": state["fc2.weight"].cpu().numpy(),
        "fc2_bias": state["fc2.bias"].cpu().numpy(),
    }
    for index in (1, 2, 3):
        block = getattr(model, f"block{index}")
        prefix = f"block{index}"
        result[f"{prefix}_dw_weight"] = block.depthwise.weight.detach().cpu().numpy()
        result[f"{prefix}_dw_bias"] = block.depthwise.bias.detach().cpu().numpy()
        result[f"{prefix}_pw_weight"] = block.pointwise.weight.detach().cpu().numpy()
        result[f"{prefix}_pw_bias"] = block.pointwise.bias.detach().cpu().numpy()
        if isinstance(block.skip, nn.Conv1d):
            result[f"{prefix}_skip_weight"] = block.skip.weight.detach().cpu().numpy()
            result[f"{prefix}_skip_bias"] = block.skip.bias.detach().cpu().numpy()
    return result


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        use_predefined=args.predefined_split,
        seed=args.seed,
    )
    classes = np.asarray(sorted(set(dataset.labels.tolist())), dtype=str)
    class_index = {label: index for index, label in enumerate(classes)}
    feature_train = add_magnitudes(train.windows)
    feature_mean = feature_train.mean(axis=(0, 1))
    feature_std = np.maximum(feature_train.std(axis=(0, 1)), 1e-6)

    train_x = torch.tensor(train.windows, dtype=torch.float32)
    validation_x = torch.tensor(validation.windows, dtype=torch.float32)
    test_x = torch.tensor(test.windows, dtype=torch.float32)
    train_y = torch.tensor([class_index[label] for label in train.labels])
    validation_y = torch.tensor([class_index[label] for label in validation.labels])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    candidates = [
        {
            "name": "balanced",
            "learning_rate": 0.0020,
            "weight_decay": 8e-4,
            "rotation_degrees": 8.0,
            "block_dropout": 0.12,
            "head_dropout": 0.25,
        },
        {
            "name": "robust",
            "learning_rate": 0.0016,
            "weight_decay": 1.2e-3,
            "rotation_degrees": 12.0,
            "block_dropout": 0.16,
            "head_dropout": 0.30,
        },
        {
            "name": "low-regularisation",
            "learning_rate": 0.0022,
            "weight_decay": 4e-4,
            "rotation_degrees": 5.0,
            "block_dropout": 0.08,
            "head_dropout": 0.18,
        },
    ]
    stress_validation_x = torch.tensor(
        stress_windows(validation.windows, args.seed + 90), dtype=torch.float32
    )
    runs: list[dict] = []
    started = time.monotonic()
    for candidate_number, config in enumerate(candidates):
        torch.manual_seed(args.seed + candidate_number)
        model = TinyTemporalCNN(
            len(classes),
            feature_mean,
            feature_std,
            block_dropout=config["block_dropout"],
            head_dropout=config["head_dropout"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.03)
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=args.batch_size,
            shuffle=True,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_selection = -np.inf
        stale = 0
        history: list[dict] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for windows, labels in loader:
                windows = random_rotation(
                    windows.to(device), config["rotation_degrees"]
                )
                windows *= 1.0 + torch.randn(
                    len(windows), 1, 6, device=device
                ) * 0.015
                windows += torch.randn_like(windows) * 0.00035
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = loss_function(model(windows), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(windows)

            validation_predicted, validation_probabilities = predictions(
                model, validation_x, device
            )
            stress_predicted, _ = predictions(model, stress_validation_x, device)
            validation_scores = scores(
                validation.labels, validation_predicted, classes
            )
            stress_scores = scores(
                validation.labels, stress_predicted, classes
            )
            selection = (
                0.65 * validation_scores["macro_f1"]
                + 0.35 * stress_scores["macro_f1"]
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_x),
                    "validation_macro_f1": validation_scores["macro_f1"],
                    "stress_validation_macro_f1": stress_scores["macro_f1"],
                }
            )
            if selection > best_selection + 1e-4:
                best_selection = selection
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        validation_predicted, _ = predictions(model, validation_x, device)
        stress_predicted, _ = predictions(model, stress_validation_x, device)
        run = {
            "config": config,
            "model": model,
            "history": history,
            "validation": scores(validation.labels, validation_predicted, classes),
            "stress_validation": scores(
                validation.labels, stress_predicted, classes
            ),
        }
        runs.append(run)
        print(
            f"{config['name']}: val F1={run['validation']['macro_f1']:.4f}, "
            f"stress F1={run['stress_validation']['macro_f1']:.4f}, "
            f"epochs={len(history)}",
            flush=True,
        )

    chosen = max(
        runs,
        key=lambda run: (
            0.65 * run["validation"]["macro_f1"]
            + 0.35 * run["stress_validation"]["macro_f1"]
        ),
    )
    model = chosen["model"]
    train_predicted, train_probabilities = predictions(model, train_x, device)
    validation_predicted, validation_probabilities = predictions(
        model, validation_x, device
    )
    test_predicted, test_probabilities = predictions(model, test_x, device)
    stress_test_x = torch.tensor(
        stress_windows(test.windows, args.seed + 91), dtype=torch.float32
    )
    stress_test_predicted, stress_test_probabilities = predictions(
        model, stress_test_x, device
    )
    train_scores = scores(train.labels, train_predicted, classes)
    validation_scores = scores(validation.labels, validation_predicted, classes)
    test_scores = scores(test.labels, test_predicted, classes)
    stress_test_scores = scores(test.labels, stress_test_predicted, classes)
    diagnosis = diagnose_fit(train_scores, validation_scores, test_scores)
    calibration_threshold = threshold_report(
        validation.labels, validation_probabilities, classes
    )
    # A classifier-only 0.5 threshold is not an adequate command gate for a
    # moving robot, even when a synthetic validation set is perfectly
    # separable.  Keep a conservative deployment floor; the continuous
    # command state machine adds temporal confirmation on top.
    recommended_threshold = max(0.85, calibration_threshold["threshold"])

    parameters = export_parameters(model)
    parameter_count = int(sum(value.size for value in parameters.values()))
    metadata = {
        "architecture": "depthwise-temporal-cnn-attentive-pooling",
        "selected_candidate": chosen["config"],
        "dataset_records": int(sum(len(session.samples) for session in sessions)),
        "split_method": split_method,
        "subject_leakage": False,
        "parameter_count": parameter_count,
        "scores": {
            "train": train_scores,
            "validation": validation_scores,
            "test": test_scores,
            "stress_test": stress_test_scores,
        },
        "fit_diagnosis": diagnosis,
        "recommended_threshold": recommended_threshold,
        "threshold_validation": calibration_threshold,
        "calibration": {
            "validation_ece": expected_calibration_error(
                validation.labels, validation_probabilities, classes
            ),
            "test_ece": expected_calibration_error(
                test.labels, test_probabilities, classes
            ),
            "stress_test_ece": expected_calibration_error(
                test.labels, stress_test_probabilities, classes
            ),
        },
        "seed": args.seed,
    }
    portable = TemporalCNNClassifier(
        classes=classes,
        target_steps=args.target_steps,
        window_seconds=args.window_seconds,
        stride_seconds=args.deployment_stride_seconds,
        feature_mean=feature_mean,
        feature_std=feature_std,
        parameters=parameters,
        metadata=metadata,
    )
    portable.save(args.output)

    # Verify that deployment NumPy inference matches the training graph.
    comparison_count = min(128, len(test.windows))
    portable_probabilities = portable.predict_proba(test.windows[:comparison_count])
    if not np.allclose(
        portable_probabilities,
        test_probabilities[:comparison_count],
        atol=2e-5,
        rtol=2e-5,
    ):
        raise RuntimeError("Portable NumPy inference does not match PyTorch")
    benchmark = np.repeat(test.windows[:1], 200, axis=0)
    benchmark_start = time.perf_counter()
    portable.predict_proba(benchmark)
    latency_ms = (
        time.perf_counter() - benchmark_start
    ) * 1000.0 / len(benchmark)
    metadata["numpy_latency_ms_per_window"] = latency_ms
    portable.metadata = metadata
    portable.save(args.output)

    report = {
        "schema": "ring-temporal-cnn-experiment/v1",
        "classes": classes.tolist(),
        "sessions": len(sessions),
        "subjects": {
            "train": len(set(train.subjects.tolist())),
            "validation": len(set(validation.subjects.tolist())),
            "test": len(set(test.subjects.tolist())),
        },
        "windows": {
            "train": len(train.windows),
            "validation": len(validation.windows),
            "test": len(test.windows),
        },
        "selected_candidate": chosen["config"],
        "candidate_validation": [
            {
                "name": run["config"]["name"],
                "validation": run["validation"],
                "stress_validation": run["stress_validation"],
                "epochs": len(run["history"]),
            }
            for run in runs
        ],
        "scores": metadata["scores"],
        "fit_diagnosis": diagnosis,
        "threshold_validation": {
            **calibration_threshold,
            "deployment_floor": recommended_threshold,
        },
        "calibration": metadata["calibration"],
        "test_confusion": confusion(test.labels, test_predicted, classes),
        "stress_test_confusion": confusion(
            test.labels, stress_test_predicted, classes
        ),
        "parameter_count": parameter_count,
        "model_bytes": args.output.stat().st_size,
        "numpy_latency_ms_per_window": latency_ms,
        "deployment_stride_seconds": args.deployment_stride_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "All class labels in this experiment are physics-informed simulation.",
            "Real captures are unlabeled and cannot establish real command accuracy.",
            "The continuous spotter and robot command state machine require separate tests.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected={chosen['config']['name']} | "
        f"test F1={test_scores['macro_f1']:.4f} | "
        f"stress F1={stress_test_scores['macro_f1']:.4f} | "
        f"parameters={parameter_count:,} | "
        f"NumPy={latency_ms:.3f} ms/window",
        flush=True,
    )


if __name__ == "__main__":
    main()
