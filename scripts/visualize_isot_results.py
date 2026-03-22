from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import OUTPUT_DIR
from src.visualize import load_model_visualization_data, render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML dashboard for ISOT baseline outputs.")
    parser.add_argument(
        "--input-dir",
        default=str(OUTPUT_DIR / "isot"),
        help="Directory containing per-model output folders.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "isot" / "dashboard.html"),
        help="Path for the generated HTML dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()

    data_items = []
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        item = load_model_visualization_data(model_dir)
        if item is not None:
            data_items.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard(data_items), encoding="utf-8")
    print(f"Saved dashboard to {output_path}")


if __name__ == "__main__":
    main()
