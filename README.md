# UQ With LLMs For Misinformation

First baseline for uncertainty quantification via self-verbalized confidence on the ISOT dataset.

## Supported models

- `gpt-4.1-mini`
- `claude-3-haiku-20240307`

## Environment variables

Create a local `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

You can copy from [.env.example](C:\Users\doghm\Desktop\projet-infox\UQ-with-LLMs-to-misinformation\.env.example). The scripts load `.env` automatically.

## Run inference + evaluation

```powershell
python scripts/run_isot_self_verbalization.py --model gpt-4.1-mini --limit 20
python scripts/run_isot_self_verbalization.py --model claude-3-haiku-20240307 --limit 20
python scripts/run_isot_self_verbalization.py --model all --limit 20
```

When `--limit` is used, the script now takes a near-balanced subset from both `True.csv` and `Fake.csv` instead of only the first `N` rows.

## Re-run evaluation on saved predictions

```powershell
python scripts/evaluate_isot_self_verbalization.py --predictions outputs/isot_baseline/gpt_4_1_mini/predictions.csv
```

Outputs are written under `outputs/isot_baseline/<model_name>/`.
