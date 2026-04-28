# Comparative Evaluation of LLMs for Structured Data Extraction from PDF Medical Laboratory Reports

CO4832 Independent Investigation — MSc Computing, UCLan Cyprus

## Overview

This system evaluates multiple large language models on the task of extracting structured data from PDF medical lab reports. Each report is processed by all configured models using an identical prompt. Outputs are compared against manually annotated ground truth using three metrics: recall, value accuracy, and hallucination rate.

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure API keys**

Copy `.env.example` to `.env` and fill in your keys:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## Usage

```bash
python main.py run        # Extract data from all PDFs using all configured models
python main.py evaluate   # Compute metrics against ground truth
python main.py summary    # Print ranked summary table
```

Run these three commands in order. The `run` step caches results — if interrupted, it resumes from where it left off. Delete a model's JSON file in `results/` to force a re-run for that model.

**Extract from a single PDF**

To run one model on one file and print the result to stdout:

```bash
python main.py extract reports_raw/sample-report-1.pdf --model gpt-4o
python main.py extract reports_raw/sample-report-1.pdf --model claude-sonnet-4-6
python main.py extract reports_raw/sample-report-1.pdf --model gemini-2.5-flash
```

## Output Format

Each model returns a JSON array of lab result objects:

```json
[
  { "label": "718-7", "value": 15.5, "unit": "g/dL" },
  { "label": "18311-0", "value": "normocytic, normochromic", "unit": null }
]
```

Labels use [LOINC codes](https://loinc.org/) as the canonical identifier. If no LOINC code exists for a test, the label falls back to `snake_case` with a specimen suffix (`_blood`, `_urine`, `_smear`).

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Recall | matched / expected | Fraction of expected fields the model found |
| Value Accuracy | correct / matched | Of found fields, fraction with correct value |
| Hallucination Rate | extra / predicted | Fraction of output fields not in ground truth |

## Adding a New Model

1. Create a new file in `src/models/` that subclasses `LLMAdapter`
2. Implement the `_call_api(self, pdf_text, prompt) -> str` method
3. Register the adapter in `get_models()` in `main.py`

No other changes are needed.
