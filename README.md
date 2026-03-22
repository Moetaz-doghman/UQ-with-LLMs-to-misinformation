# UQ With LLMs For Misinformation

First baseline for uncertainty quantification via self-verbalized confidence on the ISOT dataset.

## Supported models

- `gpt-4.1-mini`
- `claude-3-haiku-20240307`
- `gemini-1.5-flash`

## Environment variables

Create a local `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

You can copy from [.env.example](C:\Users\doghm\Desktop\projet-infox\UQ-with-LLMs-to-misinformation\.env.example). The scripts load `.env` automatically.

## Run inference + evaluation

```powershell
python scripts/run_isot_self_verbalization.py --dataset isot --model gpt-4.1-mini --limit 20
python scripts/run_isot_self_verbalization.py --dataset isot --model claude-3-haiku-20240307 --limit 20
python scripts/run_isot_self_verbalization.py --dataset isot --model gemini-1.5-flash --limit 20
python scripts/run_isot_self_verbalization.py --dataset isot --model all --limit 20
python scripts/run_isot_self_verbalization.py --dataset info-qc --model gpt-4.1-mini --limit 20
```

When `--limit` is used, the script now takes a near-balanced subset from both `True.csv` and `Fake.csv` instead of only the first `N` rows.

## Re-run evaluation on saved predictions

```powershell
python scripts/evaluate_isot_self_verbalization.py --predictions outputs/isot/gpt_4_1_mini/predictions.csv
```

## Generate visualizations

```powershell
python scripts/visualize_isot_results.py --input-dir outputs/isot --output outputs/isot/dashboard.html
python scripts/visualize_isot_results.py --input-dir outputs/info_qc --output outputs/info_qc/dashboard.html
```

This creates:
- `outputs/isot/dashboard.html`
- `outputs/info_qc/dashboard.html`

Outputs are written under `outputs/<dataset_name>/<model_name>/`.
