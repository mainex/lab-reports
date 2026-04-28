import json
import time
from pathlib import Path

from .extractor import extract_text
from .models.base import LLMAdapter

RESULTS_DIR = Path("results")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "-").replace(":", "-")


def run_all(
        models: list[LLMAdapter],
        reports_dir: Path,
        prompt: str,
        delay: float = 0.5,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    pdfs = sorted(reports_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDF files found in {reports_dir}")
        return

    for model in models:
        slug = _slug(model.model_id)
        out_path = RESULTS_DIR / f"{slug}.json"

        data: dict = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

        print(f"\n=== {model.model_id} ===")
        for pdf_path in pdfs:
            report_id = pdf_path.stem

            if report_id in data:
                print(f"  [SKIP] {report_id}")
                continue

            print(f"  {report_id} ... ", end="", flush=True)
            try:
                pdf_text = extract_text(pdf_path)
                raw = model._call_api(pdf_text, prompt)
                parsed = model._parse_response(raw)
                data[report_id] = parsed
                out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print(f"OK ({len(parsed)} fields)")
            except Exception as exc:
                print(f"ERROR: {exc}")

            time.sleep(delay)
