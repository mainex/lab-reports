import csv
import json
from pathlib import Path

_SKIP = {"summary.json"}


def _values_match(pred, truth) -> bool:
    if isinstance(truth, (int, float)) and isinstance(pred, (int, float)):
        return pred == truth if truth == 0 else abs(pred - truth) / abs(truth) < 0.001
    return str(pred).strip().lower() == str(truth).strip().lower()


def evaluate_single(predicted: list[dict], truth: list[dict]) -> dict:
    truth_by_label = {item["label"]: item for item in truth}
    pred_by_label = {item["label"]: item for item in predicted}

    truth_labels = set(truth_by_label)
    pred_labels = set(pred_by_label)
    matched = truth_labels & pred_labels

    recall = len(matched) / len(truth_labels) if truth_labels else 0.0

    correct_values = sum(
        _values_match(pred_by_label[l]["value"], truth_by_label[l]["value"])
        for l in matched
    )
    value_accuracy = correct_values / len(matched) if matched else 0.0

    hallucination_rate = len(pred_labels - truth_labels) / len(pred_labels) if pred_labels else 0.0

    return {
        "truth_count": len(truth_labels),
        "pred_count": len(pred_labels),
        "recall": round(recall, 4),
        "value_accuracy": round(value_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }


def _average(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if key in r]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def save_results(results_dir: Path, ground_truth_dir: Path) -> None:
    rows: list[dict] = []

    for model_path in sorted(results_dir.glob("*.json")):
        if model_path.name in _SKIP:
            continue
        model_slug = model_path.stem
        model_data: dict = json.loads(model_path.read_text(encoding="utf-8"))

        for report_id, predicted in model_data.items():
            num = report_id.split("-")[-1]
            gt_path = ground_truth_dir / f"expected-output-{num}.json"
            if not gt_path.exists():
                print(f"  [WARN] No ground truth for {report_id}, skipping.")
                continue
            truth = json.loads(gt_path.read_text(encoding="utf-8"))
            metrics = evaluate_single(predicted, truth)
            rows.append({"model": model_slug, "report": report_id, **metrics})

    if not rows:
        print("No results to evaluate.")
        return

    # Per-pair CSV
    csv_path = results_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # Per-model summary JSON
    models = sorted({r["model"] for r in rows})
    metric_keys = [k for k in fieldnames if k not in ("model", "report")]
    summary = {
        m: {k: _average([r for r in rows if r["model"] == m], k) for k in metric_keys}
        for m in models
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
