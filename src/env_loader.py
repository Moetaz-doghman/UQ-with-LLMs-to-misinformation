from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def inspect_env(dotenv_path: Path) -> Dict[str, object]:
    return {
        "dotenv_path": str(dotenv_path),
        "dotenv_exists": dotenv_path.exists(),
        "openai_present": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_present": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google_present": bool(os.getenv("GOOGLE_API_KEY")),
    }
