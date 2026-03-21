from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    coverage: float
    kept_count: int
    selective_accuracy: float | None
    missed_correct_rate: float


def accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    return correct / len(y_true)


def macro_f1_score(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    f1_values: List[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def roc_auc_score_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        start = idx
        score = ranked[idx][0]
        while idx < len(ranked) and ranked[idx][0] == score:
            idx += 1
        avg_rank = (start + 1 + idx) / 2.0
        rank_sum += avg_rank * sum(label for _, label in ranked[start:idx])

    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc


def compute_threshold_metrics(
    confidences: Sequence[float],
    correct_flags: Sequence[int],
    thresholds: Iterable[float] = THRESHOLDS,
) -> List[ThresholdMetrics]:
    total = len(confidences)
    total_correct = sum(correct_flags)
    metrics: List[ThresholdMetrics] = []
    for threshold in thresholds:
        kept_indices = [idx for idx, confidence in enumerate(confidences) if confidence >= threshold]
        rejected_indices = [idx for idx, confidence in enumerate(confidences) if confidence < threshold]
        kept_count = len(kept_indices)
        coverage = kept_count / total if total else 0.0
        selective_accuracy = None
        if kept_count:
            selective_accuracy = sum(correct_flags[idx] for idx in kept_indices) / kept_count
        rejected_correct = sum(correct_flags[idx] for idx in rejected_indices)
        missed_correct_rate = rejected_correct / total_correct if total_correct else 0.0
        metrics.append(
            ThresholdMetrics(
                threshold=threshold,
                coverage=coverage,
                kept_count=kept_count,
                selective_accuracy=selective_accuracy,
                missed_correct_rate=missed_correct_rate,
            )
        )
    return metrics
