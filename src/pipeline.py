from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from .config import DATA_ROOT_DIR, ExperimentConfig
from .data_loader import ArticleRecord, load_dataset, select_balanced_subset
from .evaluate import THRESHOLDS, accuracy_score, compute_threshold_metrics, macro_f1_score, roc_auc_score_binary
from .models import ModelClient
from .parser import parse_prediction
from .prompts import build_prompt, load_prompt_template


PREDICTION_FIELDNAMES = [
    "article_id",
    "source_file",
    "gold_label",
    "title",
    "text",
    "subject",
    "date",
    "pred_label",
    "confidence",
    "uncertainty",
    "justification",
    "raw_response",
    "parse_ok",
    "parse_error",
    "correct",
    "error",
]


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_predictions_csv(output_path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _append_prediction_row(output_path: Path, row: dict) -> None:
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_existing_prediction_rows(output_path: Path) -> List[dict]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_threshold_metrics_csv(output_path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["threshold", "coverage", "kept_count", "selective_accuracy", "missed_correct_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_json(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_inference(
    *,
    output_dir: Path,
    prompt_path: Path,
    model_id: str,
    config: ExperimentConfig,
    limit: int | None = None,
) -> Path:
    _ensure_directory(output_dir)
    articles = load_dataset(config.dataset_name, DATA_ROOT_DIR)
    if limit is not None:
        articles = select_balanced_subset(articles, limit)
    selected_article_ids = {article.article_id for article in articles}
    template = load_prompt_template(prompt_path)
    client = ModelClient(
        model_id=model_id,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )

    predictions_path = output_dir / "predictions.csv"
    existing_rows = [
        row for row in _load_existing_prediction_rows(predictions_path)
        if row.get("article_id") in selected_article_ids
    ]
    if predictions_path.exists():
        _write_predictions_csv(predictions_path, existing_rows)
    completed_ids = {row["article_id"] for row in existing_rows}
    prediction_rows: List[dict] = list(existing_rows)

    if completed_ids:
        print(f"Resuming run: found {len(completed_ids)} existing predictions in {predictions_path}")

    total_articles = len(articles)
    for index, article in enumerate(articles, start=1):
        if article.article_id in completed_ids:
            continue
        prompt = build_prompt(template, article, config)
        raw_response = client.generate(prompt)
        parsed = parse_prediction(raw_response)
        correct = (
            int(parsed.parse_ok and parsed.pred_label == article.gold_label)
            if parsed.parse_ok
            else 0
        )
        confidence = parsed.confidence if parsed.confidence is not None else ""
        uncertainty = (1.0 - parsed.confidence) if parsed.confidence is not None else ""
        row = {
            "article_id": article.article_id,
            "source_file": article.source_file,
            "gold_label": article.gold_label,
            "title": article.title,
            "text": article.text,
            "subject": article.subject,
            "date": article.date,
            "pred_label": parsed.pred_label or "",
            "confidence": confidence,
            "uncertainty": uncertainty,
            "justification": parsed.justification,
            "raw_response": raw_response,
            "parse_ok": int(parsed.parse_ok),
            "parse_error": parsed.parse_error,
            "correct": correct,
            "error": 1 - correct,
        }
        prediction_rows.append(row)
        _append_prediction_row(predictions_path, row)
        print(
            f"[{len(prediction_rows)}/{total_articles}] {article.article_id} "
            f"gold={article.gold_label} pred={row['pred_label'] or 'parse_failed'} "
            f"conf={row['confidence'] if row['confidence'] != '' else 'n/a'}"
        )
    return predictions_path


def evaluate_predictions(predictions_path: Path, output_dir: Path) -> None:
    _ensure_directory(output_dir)
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    valid_rows = [row for row in rows if row["parse_ok"] == "1"]
    y_true = [row["gold_label"] for row in valid_rows]
    y_pred = [row["pred_label"] for row in valid_rows]
    confidences = [float(row["confidence"]) for row in valid_rows]
    uncertainties = [float(row["uncertainty"]) for row in valid_rows]
    correct_flags = [int(row["correct"]) for row in valid_rows]
    error_flags = [int(row["error"]) for row in valid_rows]

    metrics = {
        "n_total": len(rows),
        "n_valid": len(valid_rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred, labels=["fake", "real"]),
        "uncertainty_error_roc_auc": roc_auc_score_binary(error_flags, uncertainties),
        "thresholds": THRESHOLDS,
    }
    _write_metrics_json(output_dir / "metrics.json", metrics)

    threshold_rows = [asdict(item) for item in compute_threshold_metrics(confidences, correct_flags, THRESHOLDS)]
    _write_threshold_metrics_csv(output_dir / "threshold_metrics.csv", threshold_rows)
