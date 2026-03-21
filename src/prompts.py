from __future__ import annotations

from pathlib import Path

from .data_loader import ArticleRecord


def truncate_text(text: str, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    return text[: max_characters - 16].rstrip() + "\n...[truncated]"


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_prompt(template: str, article: ArticleRecord, max_characters: int) -> str:
    return template.format(
        title=article.title,
        subject=article.subject,
        date=article.date,
        text=truncate_text(article.text, max_characters),
    )
