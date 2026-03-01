from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

KEY_TERMS = {
    "ai",
    "gpu",
    "rocm",
    "model",
    "collaboration",
    "realtime",
    "pipeline",
    "training",
    "video",
    "render",
    "accelerated",
}

FEATURE_NAMES = [
    "length_norm",
    "unique_ratio",
    "avg_word_len_norm",
    "keyword_density",
    "punctuation_ratio",
    "position_norm",
]


class LocalFineTuneTrainer:
    def __init__(self, script: str, output_dir: Path, epochs: int = 6, learning_rate: float = 0.16) -> None:
        self.script = script or ""
        self.output_dir = output_dir
        self.epochs = max(2, min(20, int(epochs)))
        self.learning_rate = float(max(0.01, min(1.0, learning_rate)))

        self.feature_names = list(FEATURE_NAMES)
        self.train_data: list[tuple[list[float], float]] = []
        self.validation_data: list[tuple[list[float], float]] = []
        self.weights = [0.0 for _ in self.feature_names]
        self.bias = 0.0
        self.dataset_meta: dict[str, Any] = {}

    def prepare(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        samples = _extract_training_samples(self.script)
        vectors: list[tuple[list[float], float, str]] = []
        for index, sample in enumerate(samples):
            features = _feature_vector(sample, index, len(samples))
            target = _target_score(sample)
            vectors.append((features, target, sample))

        split_index = max(1, int(len(vectors) * 0.8))
        if split_index >= len(vectors):
            split_index = len(vectors) - 1

        train_vectors = vectors[:split_index]
        validation_vectors = vectors[split_index:] or vectors[:1]

        self.train_data = [(features, target) for features, target, _ in train_vectors]
        self.validation_data = [(features, target) for features, target, _ in validation_vectors]

        seed = int(hashlib.sha1(self.script.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        self.weights = [rng.uniform(-0.08, 0.08) for _ in self.feature_names]
        self.bias = rng.uniform(-0.03, 0.03)

        self.dataset_meta = {
            "samples_total": len(vectors),
            "train_samples": len(self.train_data),
            "validation_samples": len(self.validation_data),
            "feature_names": list(self.feature_names),
        }
        return dict(self.dataset_meta)

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        feature_count = len(self.weights)
        gradient_w = [0.0 for _ in range(feature_count)]
        gradient_b = 0.0
        loss_total = 0.0

        for features, target in self.train_data:
            logit = _dot(self.weights, features) + self.bias
            prediction = _sigmoid(logit)
            error = prediction - target
            for feature_index in range(feature_count):
                gradient_w[feature_index] += error * features[feature_index]
            gradient_b += error
            loss_total += _binary_cross_entropy(prediction, target)

        sample_scale = 1.0 / max(len(self.train_data), 1)
        for feature_index in range(feature_count):
            self.weights[feature_index] -= self.learning_rate * gradient_w[feature_index] * sample_scale
        self.bias -= self.learning_rate * gradient_b * sample_scale

        train_loss = loss_total * sample_scale
        validation_loss = self._evaluate(self.validation_data)
        weight_norm = math.sqrt(sum(weight * weight for weight in self.weights))

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:02d}.json"
        checkpoint_payload = {
            "epoch": epoch,
            "learning_rate": self.learning_rate,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation_loss, 6),
            "weight_norm": round(weight_norm, 6),
            "weights": [round(weight, 8) for weight in self.weights],
            "bias": round(self.bias, 8),
        }
        checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")

        return {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(validation_loss, 4),
            "weight_norm": round(weight_norm, 4),
            "checkpoint_path": checkpoint_path,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "feature_names": list(self.feature_names),
            "final_weights": [round(weight, 6) for weight in self.weights],
            "final_bias": round(self.bias, 6),
            "final_validation_loss": round(self._evaluate(self.validation_data), 4),
        }

    def _evaluate(self, dataset: list[tuple[list[float], float]]) -> float:
        if not dataset:
            return 0.0

        loss_sum = 0.0
        for features, target in dataset:
            prediction = _sigmoid(_dot(self.weights, features) + self.bias)
            loss_sum += _binary_cross_entropy(prediction, target)
        return loss_sum / len(dataset)


def _extract_training_samples(script: str) -> list[str]:
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not lines:
        lines = _sentence_split(script)

    if not lines:
        lines = [
            "Team collaborates on script generation.",
            "AI creates visual scenes and narration.",
            "Output is rendered into a demo package.",
        ]

    samples: list[str] = []
    for index, line in enumerate(lines):
        samples.append(line)
        if index + 1 < len(lines):
            samples.append(f"{line} {lines[index + 1]}")

    while len(samples) < 8:
        source = lines[len(samples) % len(lines)]
        samples.append(f"{source} Collaborative AI acceleration pipeline.")

    return samples[:18]


def _feature_vector(text: str, index: int, total: int) -> list[float]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    word_count = len(words)
    unique_ratio = len(set(words)) / word_count if word_count else 0.0
    avg_word_len = sum(len(word) for word in words) / word_count if word_count else 0.0
    keyword_hits = sum(1 for word in words if word in KEY_TERMS)
    keyword_density = keyword_hits / max(word_count, 1)
    punctuation_ratio = len(re.findall(r"[,.!?;:]", text)) / max(len(text), 1)
    position_norm = index / max(total - 1, 1)

    return [
        _clamp(word_count / 24.0, 0.0, 1.0),
        _clamp(unique_ratio, 0.0, 1.0),
        _clamp(avg_word_len / 8.0, 0.0, 1.0),
        _clamp(keyword_density * 4.0, 0.0, 1.0),
        _clamp(punctuation_ratio * 10.0, 0.0, 1.0),
        _clamp(position_norm, 0.0, 1.0),
    ]


def _target_score(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)
    keyword_hits = sum(1 for word in words if word in KEY_TERMS)
    keyword_density = keyword_hits / max(word_count, 1)
    punctuation = len(re.findall(r"[,.!?;:]", text))

    score = (
        0.16
        + 0.34 * _clamp(keyword_density * 4.0, 0.0, 1.0)
        + 0.22 * _clamp(word_count / 24.0, 0.0, 1.0)
        + 0.18 * _clamp(unique_ratio, 0.0, 1.0)
        + 0.10 * _clamp(punctuation / 5.0, 0.0, 1.0)
    )
    return _clamp(score, 0.05, 0.95)


def _sentence_split(text: str) -> list[str]:
    collapsed = " ".join(text.replace("\n", " ").split())
    if not collapsed:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", collapsed) if segment.strip()]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _binary_cross_entropy(prediction: float, target: float) -> float:
    clipped = _clamp(prediction, 1e-6, 1.0 - 1e-6)
    return -((target * math.log(clipped)) + ((1.0 - target) * math.log(1.0 - clipped)))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
