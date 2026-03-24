from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List


COLORS = {
    "correct": "#2a9d8f",
    "wrong": "#e76f51",
    "line": "#264653",
    "accent": "#e9c46a",
    "grid": "#d9d9d9",
    "text": "#1f2933",
}


@dataclass(frozen=True)
class ModelVisualizationData:
    model_name: str
    metrics: dict
    threshold_rows: list[dict]
    predictions: list[dict]


def load_model_visualization_data(model_dir: Path) -> ModelVisualizationData | None:
    metrics_path = model_dir / "metrics.json"
    threshold_path = model_dir / "threshold_metrics.csv"
    predictions_path = model_dir / "predictions.csv"
    if not (metrics_path.exists() and threshold_path.exists() and predictions_path.exists()):
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with threshold_path.open("r", encoding="utf-8", newline="") as handle:
        threshold_rows = list(csv.DictReader(handle))
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        predictions = [row for row in csv.DictReader(handle) if row.get("parse_ok") == "1"]

    return ModelVisualizationData(
        model_name=model_dir.name,
        metrics=metrics,
        threshold_rows=threshold_rows,
        predictions=predictions,
    )


def render_dashboard(data_items: Iterable[ModelVisualizationData]) -> str:
    data_items = list(data_items)
    body = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>ISOT Baseline Dashboard</title>",
        _styles(),
        "</head>",
        "<body>",
        "<main>",
        "<h1>ISOT Self-Verbalization Dashboard</h1>",
        "<p class='lede'>Interpretation view for accuracy, uncertainty quality, and selective prediction tradeoffs.</p>",
    ]

    if not data_items:
        body.append("<p>No completed model outputs were found.</p>")
    else:
        body.append("<section>")
        body.append("<h2>Model Summary</h2>")
        body.append(_render_summary_table(data_items))
        body.append("</section>")
        for item in data_items:
            body.append("<section class='model-section'>")
            body.append(f"<h2>{html.escape(item.model_name)}</h2>")
            body.append(_render_metric_cards(item))
            body.append("<div class='chart-grid'>")
            body.append(_render_roc_chart(item))
            body.append(_render_threshold_chart(item))
            body.append(_render_histogram_chart(item))
            body.append("</div>")
            body.append(_render_interpretation(item))
            body.append("</section>")

    body.extend(["</main>", "</body>", "</html>"])
    return "\n".join(body)


def _styles() -> str:
    return """
<style>
  :root {
    --bg: #f7f3eb;
    --panel: #fffdf8;
    --ink: #1f2933;
    --muted: #5c6773;
    --line: #d9d9d9;
    --correct: #2a9d8f;
    --wrong: #e76f51;
    --accent: #e9c46a;
    --chart: #264653;
  }
  body { margin: 0; font-family: Georgia, 'Times New Roman', serif; color: var(--ink); background: linear-gradient(180deg, #f7f3eb 0%, #efe6d7 100%); }
  main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }
  h1, h2, h3 { margin: 0 0 12px; }
  .lede { color: var(--muted); max-width: 820px; margin-bottom: 24px; }
  section { background: var(--panel); border: 1px solid #eadfcd; border-radius: 18px; padding: 20px; margin-bottom: 22px; box-shadow: 0 8px 24px rgba(38, 70, 83, 0.06); }
  .summary-table { width: 100%; border-collapse: collapse; }
  .summary-table th, .summary-table td { border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; }
  .summary-table th { font-size: 0.92rem; color: var(--muted); }
  .metric-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 18px; }
  .card { background: #fff8ea; border: 1px solid #eadfcd; border-radius: 14px; padding: 14px; }
  .card .label { font-size: 0.84rem; color: var(--muted); margin-bottom: 6px; }
  .card .value { font-size: 1.5rem; font-weight: 700; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; align-items: start; }
  .chart { background: #fff; border: 1px solid #efe3d1; border-radius: 14px; padding: 14px; }
  .chart p { color: var(--muted); margin: 6px 0 0; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; color: var(--muted); font-size: 0.9rem; }
  .legend span::before { content: ''; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  .legend .correct::before { background: var(--correct); }
  .legend .wrong::before { background: var(--wrong); }
  .legend .line::before { background: var(--chart); }
  .legend .accent::before { background: var(--accent); }
  .interpretation { margin-top: 14px; color: var(--ink); }
  .interpretation p { margin: 6px 0; }
</style>
"""


def _render_summary_table(data_items: list[ModelVisualizationData]) -> str:
    rows = [
        "<table class='summary-table'>",
        "<thead><tr><th>Model</th><th>Valid / Total</th><th>Accuracy</th><th>Macro-F1</th><th>Uncertainty-Error AUC</th><th>Avg Confidence (Correct)</th><th>Avg Confidence (Wrong)</th></tr></thead>",
        "<tbody>",
    ]
    for item in data_items:
        correct_conf = _confidence_values(item.predictions, correct=1)
        wrong_conf = _confidence_values(item.predictions, correct=0)
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.model_name)}</td>"
            f"<td>{item.metrics.get('n_valid', 0)} / {item.metrics.get('n_total', 0)}</td>"
            f"<td>{_fmt(item.metrics.get('accuracy'))}</td>"
            f"<td>{_fmt(item.metrics.get('macro_f1'))}</td>"
            f"<td>{_fmt(item.metrics.get('uncertainty_error_roc_auc'))}</td>"
            f"<td>{_fmt(mean(correct_conf) if correct_conf else None)}</td>"
            f"<td>{_fmt(mean(wrong_conf) if wrong_conf else None)}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _render_metric_cards(item: ModelVisualizationData) -> str:
    cards = [
        ("Valid rows", f"{item.metrics.get('n_valid', 0)} / {item.metrics.get('n_total', 0)}"),
        ("Accuracy", _fmt(item.metrics.get("accuracy"))),
        ("Macro-F1", _fmt(item.metrics.get("macro_f1"))),
        ("Uncertainty AUC", _fmt(item.metrics.get("uncertainty_error_roc_auc"))),
    ]
    values = item.predictions
    correct_conf = _confidence_values(values, correct=1)
    wrong_conf = _confidence_values(values, correct=0)
    cards.append(("Avg confidence if correct", _fmt(mean(correct_conf) if correct_conf else None)))
    cards.append(("Avg confidence if wrong", _fmt(mean(wrong_conf) if wrong_conf else None)))
    return "<div class='metric-cards'>" + "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    ) + "</div>"


def _render_threshold_chart(item: ModelVisualizationData) -> str:
    thresholds = [float(row["threshold"]) for row in item.threshold_rows]
    coverage = [float(row["coverage"]) for row in item.threshold_rows]
    selective_accuracy = [
        float(row["selective_accuracy"]) if row["selective_accuracy"] else 0.0
        for row in item.threshold_rows
    ]
    missed = [float(row["missed_correct_rate"]) for row in item.threshold_rows]
    svg = _multi_line_svg(
        x_values=thresholds,
        series=[
            ("Coverage", coverage, COLORS["line"]),
            ("Selective accuracy", selective_accuracy, COLORS["correct"]),
            ("Missed correct rate", missed, COLORS["wrong"]),
        ],
        x_label="Confidence threshold",
    )
    return (
        "<div class='chart'>"
        "<h3>Threshold Tradeoff</h3>"
        f"{svg}"
        "<div class='legend'>"
        "<span class='line'>Coverage</span>"
        "<span class='correct'>Selective accuracy</span>"
        "<span class='wrong'>Missed correct rate</span>"
        "</div>"
        "<p>This chart shows what you gain in reliability and what you lose in coverage as the confidence threshold increases.</p>"
        "</div>"
    )


def _render_roc_chart(item: ModelVisualizationData) -> str:
    roc_points = _roc_curve_points(item.predictions)
    auc = item.metrics.get("uncertainty_error_roc_auc")
    svg = _roc_svg(roc_points)
    description = (
        "This is the real ROC curve for error detection using uncertainty. "
        "The x-axis is the false positive rate and the y-axis is the true positive rate."
    )
    return (
        "<div class='chart'>"
        "<h3>ROC Curve</h3>"
        f"{svg}"
        "<div class='legend'>"
        "<span class='line'>ROC curve</span>"
        "<span class='accent'>Random baseline</span>"
        "</div>"
        f"<p>{html.escape(description)} AUC = {_fmt(auc)}.</p>"
        "</div>"
    )


def _render_histogram_chart(item: ModelVisualizationData) -> str:
    correct_conf = _confidence_values(item.predictions, correct=1)
    wrong_conf = _confidence_values(item.predictions, correct=0)
    svg = _histogram_svg(correct_conf, wrong_conf)
    return (
        "<div class='chart'>"
        "<h3>Confidence Distribution</h3>"
        f"{svg}"
        "<div class='legend'>"
        "<span class='correct'>Correct predictions</span>"
        "<span class='wrong'>Wrong predictions</span>"
        "</div>"
        "<p>If the wrong histogram shifts left and the correct histogram shifts right, the confidence signal is informative.</p>"
        "</div>"
    )


def _render_interpretation(item: ModelVisualizationData) -> str:
    auc = item.metrics.get("uncertainty_error_roc_auc")
    accuracy = item.metrics.get("accuracy")
    correct_conf = _confidence_values(item.predictions, correct=1)
    wrong_conf = _confidence_values(item.predictions, correct=0)
    avg_correct = mean(correct_conf) if correct_conf else None
    avg_wrong = mean(wrong_conf) if wrong_conf else None

    if auc is None:
        auc_text = "AUC is unavailable because there were not enough both correct and wrong predictions."
    elif auc >= 0.8:
        auc_text = "The uncertainty signal is strong: wrong predictions are usually more uncertain than correct ones."
    elif auc >= 0.65:
        auc_text = "The uncertainty signal is useful but not very sharp."
    elif auc >= 0.5:
        auc_text = "The uncertainty signal is weak: it separates errors only slightly better than chance."
    else:
        auc_text = "The uncertainty signal is poor or inverted."

    conf_text = "Average confidence on correct predictions is "
    conf_text += _fmt(avg_correct)
    conf_text += " versus "
    conf_text += _fmt(avg_wrong)
    conf_text += " on wrong predictions."

    accuracy_text = f"Overall classification accuracy is {_fmt(accuracy)}."
    return (
        "<div class='interpretation'>"
        f"<p>{html.escape(accuracy_text)}</p>"
        f"<p>{html.escape(auc_text)}</p>"
        f"<p>{html.escape(conf_text)}</p>"
        "</div>"
    )


def _confidence_values(predictions: list[dict], correct: int) -> list[float]:
    values: List[float] = []
    for row in predictions:
        if int(row["correct"]) == correct:
            values.append(float(row["confidence"]))
    return values


def _roc_curve_points(predictions: list[dict]) -> list[tuple[float, float]]:
    scored = sorted(
        ((float(row["uncertainty"]), int(row["error"])) for row in predictions),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(label for _, label in scored)
    negatives = len(scored) - positives
    if positives == 0 or negatives == 0:
        return [(0.0, 0.0), (1.0, 1.0)]

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    tp = 0
    fp = 0
    idx = 0
    while idx < len(scored):
        score = scored[idx][0]
        while idx < len(scored) and scored[idx][0] == score:
            if scored[idx][1] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        points.append((fpr, tpr))

    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))
    return points


def _multi_line_svg(x_values: list[float], series: list[tuple[str, list[float], str]], x_label: str) -> str:
    width, height = 520, 260
    left, top, right, bottom = 48, 16, 18, 36
    chart_w = width - left - right
    chart_h = height - top - bottom
    x_min, x_max = min(x_values), max(x_values)

    def project_x(value: float) -> float:
        if x_max == x_min:
            return left
        return left + (value - x_min) / (x_max - x_min) * chart_w

    def project_y(value: float) -> float:
        return top + (1 - value) * chart_h

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='threshold tradeoff chart'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' rx='12'/>",
    ]
    for step in range(6):
        y = top + step * chart_h / 5
        label = f"{1 - step / 5:.1f}"
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + chart_w}' y2='{y:.1f}' stroke='{COLORS['grid']}' stroke-width='1'/>")
        parts.append(f"<text x='10' y='{y + 4:.1f}' font-size='11' fill='{COLORS['text']}'>{label}</text>")

    for threshold in x_values:
        x = project_x(threshold)
        parts.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + chart_h}' stroke='{COLORS['grid']}' stroke-width='1'/>")
        parts.append(f"<text x='{x - 10:.1f}' y='{height - 10}' font-size='11' fill='{COLORS['text']}'>{threshold:.1f}</text>")

    for _, values, color in series:
        points = " ".join(f"{project_x(x):.1f},{project_y(y):.1f}" for x, y in zip(x_values, values))
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}'/>")

    parts.append(f"<text x='{width / 2 - 45:.1f}' y='{height - 2}' font-size='12' fill='{COLORS['text']}'>{html.escape(x_label)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _histogram_svg(correct_conf: list[float], wrong_conf: list[float]) -> str:
    bins = [i / 10 for i in range(11)]
    correct_counts = _bin_counts(correct_conf, bins)
    wrong_counts = _bin_counts(wrong_conf, bins)
    max_count = max(correct_counts + wrong_counts + [1])

    width, height = 520, 260
    left, top, right, bottom = 40, 16, 18, 36
    chart_w = width - left - right
    chart_h = height - top - bottom
    group_width = chart_w / 10
    bar_width = group_width * 0.36

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='confidence histogram'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' rx='12'/>",
    ]

    for step in range(6):
        y = top + step * chart_h / 5
        label = int(round(max_count * (1 - step / 5)))
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + chart_w}' y2='{y:.1f}' stroke='{COLORS['grid']}' stroke-width='1'/>")
        parts.append(f"<text x='6' y='{y + 4:.1f}' font-size='11' fill='{COLORS['text']}'>{label}</text>")

    for idx in range(10):
        x0 = left + idx * group_width
        c_height = (correct_counts[idx] / max_count) * chart_h
        w_height = (wrong_counts[idx] / max_count) * chart_h
        parts.append(
            f"<rect x='{x0 + group_width * 0.10:.1f}' y='{top + chart_h - c_height:.1f}' width='{bar_width:.1f}' height='{c_height:.1f}' fill='{COLORS['correct']}' rx='2'/>"
        )
        parts.append(
            f"<rect x='{x0 + group_width * 0.54:.1f}' y='{top + chart_h - w_height:.1f}' width='{bar_width:.1f}' height='{w_height:.1f}' fill='{COLORS['wrong']}' rx='2'/>"
        )
        parts.append(f"<text x='{x0 + group_width * 0.34:.1f}' y='{height - 10}' font-size='11' fill='{COLORS['text']}'>{idx/10:.1f}</text>")

    parts.append(f"<text x='{width / 2 - 50:.1f}' y='{height - 2}' font-size='12' fill='{COLORS['text']}'>Confidence bins</text>")
    parts.append("</svg>")
    return "".join(parts)


def _roc_svg(points: list[tuple[float, float]]) -> str:
    width, height = 520, 260
    left, top, right, bottom = 40, 16, 18, 36
    chart_w = width - left - right
    chart_h = height - top - bottom

    def project_x(value: float) -> float:
        return left + value * chart_w

    def project_y(value: float) -> float:
        return top + (1 - value) * chart_h

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='roc curve'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' rx='12'/>",
    ]

    for step in range(6):
        value = step / 5
        x = project_x(value)
        y = project_y(value)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + chart_w}' y2='{y:.1f}' stroke='{COLORS['grid']}' stroke-width='1'/>")
        parts.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + chart_h}' stroke='{COLORS['grid']}' stroke-width='1'/>")
        parts.append(f"<text x='{x - 8:.1f}' y='{height - 10}' font-size='11' fill='{COLORS['text']}'>{value:.1f}</text>")
        parts.append(f"<text x='8' y='{y + 4:.1f}' font-size='11' fill='{COLORS['text']}'>{1 - value:.1f}</text>")

    parts.append(
        f"<line x1='{left}' y1='{top + chart_h}' x2='{left + chart_w}' y2='{top}' stroke='{COLORS['accent']}' stroke-width='2' stroke-dasharray='6 6'/>"
    )
    roc_points = " ".join(f"{project_x(x):.1f},{project_y(y):.1f}" for x, y in points)
    parts.append(f"<polyline fill='none' stroke='{COLORS['line']}' stroke-width='3' points='{roc_points}'/>")
    parts.append(f"<text x='{width / 2 - 40:.1f}' y='{height - 2}' font-size='12' fill='{COLORS['text']}'>False positive rate</text>")
    parts.append(f"<text x='12' y='12' font-size='12' fill='{COLORS['text']}'>True positive rate</text>")
    parts.append("</svg>")
    return "".join(parts)


def _bin_counts(values: list[float], bins: list[float]) -> list[int]:
    counts = [0] * (len(bins) - 1)
    for value in values:
        idx = min(int(value * 10), 9)
        counts[idx] += 1
    return counts


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"
