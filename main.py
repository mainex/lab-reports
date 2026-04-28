"""
Entry point for the lab report LLM evaluation system.

Usage:
  python main.py run                              # Extract data from all PDFs using all configured models
  python main.py evaluate                         # Compute metrics from saved results vs. ground truth
  python main.py summary                          # Print a ranked summary table to stdout
  python main.py extract <pdf> --model <model_id> # Extract from a single PDF and print JSON to stdout
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPORTS_DIR     = Path("reports_raw")
GROUND_TRUTH_DIR = Path("ground_truth")
RESULTS_DIR     = Path("results")
PROMPT_FILE     = Path("prompt")


# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------

def get_models():
    """Return all adapters whose API key is present in the environment."""
    models = []

    if os.environ.get("OPENAI_API_KEY"):
        from src.models.openai_adapter import OpenAIAdapter
        models.append(OpenAIAdapter("gpt-4o-mini"))
        models.append(OpenAIAdapter("gpt-4o"))

    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.models.anthropic_adapter import AnthropicAdapter
        models.append(AnthropicAdapter("claude-haiku-4-5-20251001"))
        models.append(AnthropicAdapter("claude-sonnet-4-6"))

    if os.environ.get("GOOGLE_API_KEY"):
        from src.models.gemini_adapter import GeminiAdapter
        models.append(GeminiAdapter("gemini-2.5-flash"))

    return models


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run():
    from src.runner import run_all

    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    models = get_models()

    if not models:
        print(
            "No API keys found. Copy .env.example to .env and fill in your keys.\n"
            "  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
        )
        sys.exit(1)

    print(f"Running {len(models)} model(s) on {len(list(REPORTS_DIR.glob('*.pdf')))} report(s).")
    run_all(models, REPORTS_DIR, prompt)
    print("\nDone. Run 'python main.py evaluate' next.")


def cmd_evaluate():
    from src.evaluator import save_results

    model_files = [f for f in RESULTS_DIR.glob("*.json") if f.name != "summary.json"]
    if not RESULTS_DIR.exists() or not model_files:
        print("No results found. Run 'python main.py run' first.")
        sys.exit(1)

    save_results(RESULTS_DIR, GROUND_TRUTH_DIR)
    print("\nDone. Run 'python main.py summary' to view results.")


def cmd_summary():
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        print("No summary found. Run 'python main.py evaluate' first.")
        sys.exit(1)

    data: dict[str, dict] = json.loads(summary_path.read_text())

    header = f"{'Model':<42} {'Recall':>8} {'Val Acc':>8} {'Halluc.':>9}"
    print("\n" + header)
    print("-" * len(header))

    for model, m in sorted(data.items(), key=lambda x: -x[1].get("recall", 0)):
        print(
            f"{model:<42} "
            f"{m.get('recall', 0):>8.3f} "
            f"{m.get('value_accuracy', 0):>8.3f} "
            f"{m.get('hallucination_rate', 0):>9.3f}"
        )
    print()


def cmd_extract():
    """Extract from a single PDF with a single model and print JSON to stdout."""
    args = sys.argv[2:]
    if "--model" not in args or len(args) < 3:
        print("Usage: python main.py extract <pdf_path> --model <model_id>")
        print("Example: python main.py extract reports_raw/sample-report-1.pdf --model gpt-4o")
        sys.exit(1)

    model_idx = args.index("--model")
    pdf_path = Path(args[0])
    model_id = args[model_idx + 1]

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    all_models = get_models()
    model = next((m for m in all_models if m.model_id == model_id), None)
    if model is None:
        available = [m.model_id for m in all_models]
        print(f"Model '{model_id}' not found or API key not set.")
        print(f"Available models: {', '.join(available)}")
        sys.exit(1)

    from src.extractor import extract_text
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    pdf_text = extract_text(pdf_path)
    result = model.extract(pdf_text, prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

COMMANDS = {"run": cmd_run, "evaluate": cmd_evaluate, "summary": cmd_summary, "extract": cmd_extract}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(__doc__)
