from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .data_loader import ArticleRecord


def truncate_text(text: str, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    return text[: max_characters - 16].rstrip() + "\n...[truncated]"


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_prompt(template: str, article: ArticleRecord, config: ExperimentConfig) -> str:
    if config.dataset_name.strip().lower() in {"info-qc", "info_qc"}:
        return (
            "You are classifying a news-related claim or information snippet as fake or real for a misinformation detection research experiment.\n\n"
            "Task:\n"
            "1. Predict whether the content is fake or real.\n"
            "2. Report your confidence as a number between 0 and 1.\n"
            "3. Give a short justification in 1-2 sentences.\n\n"
            "Definitions:\n"
            "- fake = misinformation, fabricated, deceptive, scam-like, or strongly misleading content\n"
            "- real = legitimate factual information or truthful reporting\n\n"
            "Return your answer as valid JSON with exactly these keys:\n"
            "{\n"
            '  "label": "fake" or "real",\n'
            '  "confidence": <float between 0 and 1>,\n'
            '  "justification": "<short explanation>"\n'
            "}\n\n"
            "Do not include markdown fences. Do not include any extra keys or extra text.\n\n"
            "Content:\n"
            f"{truncate_text(article.text, config.max_characters)}"
        )

    return template.format(
        title=article.title,
        subject=article.subject,
        date=article.date,
        text=truncate_text(article.text, config.max_characters),
    )
