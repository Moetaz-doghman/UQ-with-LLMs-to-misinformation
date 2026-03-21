from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    source_file: str
    gold_label: str
    title: str
    text: str
    subject: str
    date: str


LABEL_BY_FILENAME = {
    "True.csv": "real",
    "Fake.csv": "fake",
}


def _load_csv_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def load_isot_dataset(data_dir: Path) -> List[ArticleRecord]:
    records: List[ArticleRecord] = []
    for file_name, gold_label in LABEL_BY_FILENAME.items():
        csv_path = data_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")
        for idx, row in enumerate(_load_csv_rows(csv_path)):
            records.append(
                ArticleRecord(
                    article_id=f"{csv_path.stem.lower()}-{idx}",
                    source_file=file_name,
                    gold_label=gold_label,
                    title=(row.get("title") or "").strip(),
                    text=(row.get("text") or "").strip(),
                    subject=(row.get("subject") or "").strip(),
                    date=(row.get("date") or "").strip(),
                )
            )
    return records


def select_balanced_subset(records: List[ArticleRecord], limit: int) -> List[ArticleRecord]:
    if limit <= 0 or limit >= len(records):
        return records[:limit] if limit > 0 else []

    by_label = {
        "real": [record for record in records if record.gold_label == "real"],
        "fake": [record for record in records if record.gold_label == "fake"],
    }

    real_target = (limit + 1) // 2
    fake_target = limit // 2
    selected_real = by_label["real"][:real_target]
    selected_fake = by_label["fake"][:fake_target]

    # If one class cannot fill its share, backfill from the other class.
    if len(selected_real) < real_target:
        missing = real_target - len(selected_real)
        selected_fake = by_label["fake"][: fake_target + missing]
    if len(selected_fake) < fake_target:
        missing = fake_target - len(selected_fake)
        selected_real = by_label["real"][: real_target + missing]

    balanced: List[ArticleRecord] = []
    max_len = max(len(selected_real), len(selected_fake))
    for idx in range(max_len):
        if idx < len(selected_real):
            balanced.append(selected_real[idx])
        if idx < len(selected_fake):
            balanced.append(selected_fake[idx])
        if len(balanced) >= limit:
            break
    return balanced[:limit]
