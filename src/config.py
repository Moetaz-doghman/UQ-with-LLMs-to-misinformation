from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "isot"
OUTPUT_DIR = ROOT_DIR / "outputs"
PROMPTS_DIR = ROOT_DIR / "prompts"
DOTENV_PATH = ROOT_DIR / ".env"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "isot"
    max_characters: int = 6000
    temperature: float = 0.0
    timeout_seconds: int = 120
