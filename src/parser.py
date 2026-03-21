from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


VALID_LABELS = {"fake", "real"}


@dataclass(frozen=True)
class ParsedPrediction:
    pred_label: Optional[str]
    confidence: Optional[float]
    justification: str
    parse_ok: bool
    parse_error: str


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw_text: str) -> str:
    match = JSON_BLOCK_PATTERN.search(raw_text)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return match.group(0)


def parse_prediction(raw_text: str) -> ParsedPrediction:
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except (ValueError, json.JSONDecodeError) as exc:
        return ParsedPrediction(
            pred_label=None,
            confidence=None,
            justification="",
            parse_ok=False,
            parse_error=str(exc),
        )

    label = str(payload.get("label", "")).strip().lower()
    if label not in VALID_LABELS:
        return ParsedPrediction(
            pred_label=None,
            confidence=None,
            justification=str(payload.get("justification", "")).strip(),
            parse_ok=False,
            parse_error=f"Invalid label: {label!r}",
        )

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return ParsedPrediction(
            pred_label=label,
            confidence=None,
            justification=str(payload.get("justification", "")).strip(),
            parse_ok=False,
            parse_error="Confidence is not a valid float.",
        )

    if not 0.0 <= confidence <= 1.0:
        return ParsedPrediction(
            pred_label=label,
            confidence=None,
            justification=str(payload.get("justification", "")).strip(),
            parse_ok=False,
            parse_error="Confidence must be between 0 and 1.",
        )

    return ParsedPrediction(
        pred_label=label,
        confidence=confidence,
        justification=str(payload.get("justification", "")).strip(),
        parse_ok=True,
        parse_error="",
    )
