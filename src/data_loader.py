from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from xml.etree import ElementTree as ET
from zipfile import ZipFile


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

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _load_csv_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row

# load the isot dataset and creates one list of ArticleRecord  
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


def _xlsx_cell_column(cell_ref: str) -> str:
    letters = []
    for char in cell_ref:
        if char.isalpha():
            letters.append(char)
        else:
            break
    return "".join(letters)


def _read_xlsx_rows(xlsx_path: Path) -> List[dict[str, str]]:
    with ZipFile(xlsx_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{XLSX_NS}si"):
                text_chunks = [node.text or "" for node in item.iter(f"{XLSX_NS}t")]
                shared_strings.append("".join(text_chunks))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        row_values: list[dict[str, str]] = []
        for row in sheet_root.findall(f".//{XLSX_NS}row"):
            values: dict[str, str] = {}
            for cell in row.findall(f"{XLSX_NS}c"):
                column = _xlsx_cell_column(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{XLSX_NS}v")
                value = ""
                if cell_type == "s" and value_node is not None and value_node.text is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr":
                    text_node = cell.find(f".//{XLSX_NS}t")
                    value = text_node.text if text_node is not None and text_node.text is not None else ""
                elif value_node is not None and value_node.text is not None:
                    value = value_node.text
                values[column] = value
            row_values.append(values)

    if not row_values:
        return []
    header = row_values[0]
    ordered_columns = sorted(header.keys())
    field_names = [header.get(column, "").strip() for column in ordered_columns]
    records: list[dict[str, str]] = []
    for row in row_values[1:]:
        record: dict[str, str] = {}
        for column, field_name in zip(ordered_columns, field_names):
            if field_name:
                record[field_name] = (row.get(column) or "").strip()
        records.append(record)
    return records


def load_info_qc_dataset(data_dir: Path) -> List[ArticleRecord]:
    records: List[ArticleRecord] = []
    file_specs = [
        (data_dir / "true" / "v1.xlsx", "real"),
        (data_dir / "false" / "v3.xlsx", "fake"),
    ]
    for xlsx_path, gold_label in file_specs:
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {xlsx_path}")
        for idx, row in enumerate(_read_xlsx_rows(xlsx_path)):
            title = (row.get("type_d_infox") or "").strip()
            text = (row.get("contenu_de_l_info") or "").strip()
            subject = (row.get("main_theme") or row.get("main theme") or row.get("thématique") or "").strip()
            date = (row.get("date_de_repérage") or "").strip()
            source_id = (row.get("id") or str(idx)).strip()
            records.append(
                ArticleRecord(
                    article_id=f"infoqc-{gold_label}-{source_id}",
                    source_file=str(xlsx_path.relative_to(data_dir)),
                    gold_label=gold_label,
                    title=title,
                    text=text,
                    subject=subject,
                    date=date,
                )
            )
    return records


def load_dataset(dataset_name: str, data_root_dir: Path) -> List[ArticleRecord]:
    normalized = dataset_name.strip().lower()
    if normalized == "isot":
        return load_isot_dataset(data_root_dir / "isot")
    if normalized in {"info-qc", "info_qc"}:
        return load_info_qc_dataset(data_root_dir / "info-QC")
    raise ValueError(f"Unsupported dataset: {dataset_name}")

#takes a near-balanced subset from both classes used when --limit is passed 
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
