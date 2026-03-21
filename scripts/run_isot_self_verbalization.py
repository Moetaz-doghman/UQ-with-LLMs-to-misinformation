from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, DOTENV_PATH, OUTPUT_DIR, PROMPTS_DIR, ExperimentConfig
from src.env_loader import inspect_env, load_dotenv
from src.pipeline import evaluate_predictions, run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISOT self-verbalization baseline.")
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=["gpt-4.1-mini", "claude-3-haiku-20240307", "gemini-1.5-flash", "all"],
        help="One or more model backends to use, or 'all'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of articles for quick runs.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=6000,
        help="Max characters of article text to include in the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature sent to the model API.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout for model API calls.",
    )
    parser.add_argument(
        "--debug-env",
        action="store_true",
        help="Print .env loading diagnostics before running inference.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(DOTENV_PATH)
    args = parse_args()
    if args.debug_env:
        diagnostics = inspect_env(DOTENV_PATH)
        print(diagnostics)
    config = ExperimentConfig(
        max_characters=args.max_characters,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    model_ids = (
        ["gpt-4.1-mini", "claude-3-haiku-20240307", "gemini-1.5-flash"]
        if "all" in args.model
        else args.model
    )
    for model_id in model_ids:
        model_dir_name = model_id.replace(".", "_").replace("-", "_")
        run_output_dir = OUTPUT_DIR / "isot_baseline" / model_dir_name
        predictions_path = run_inference(
            data_dir=DATA_DIR,
            output_dir=run_output_dir,
            prompt_path=PROMPTS_DIR / "self_verbalization_isot.txt",
            model_id=model_id,
            config=config,
            limit=args.limit,
        )
        evaluate_predictions(predictions_path, run_output_dir)
        print(f"Saved predictions and metrics to {run_output_dir}")


if __name__ == "__main__":
    main()
