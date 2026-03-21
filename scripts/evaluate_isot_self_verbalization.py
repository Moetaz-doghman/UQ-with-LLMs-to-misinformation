from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DOTENV_PATH
from src.env_loader import load_dotenv
from src.pipeline import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved ISOT predictions.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for metrics outputs. Defaults to the predictions parent directory.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(DOTENV_PATH)
    args = parse_args()
    predictions_path = Path(args.predictions).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else predictions_path.parent
    evaluate_predictions(predictions_path, output_dir)
    print(f"Saved metrics to {output_dir}")


if __name__ == "__main__":
    main()
