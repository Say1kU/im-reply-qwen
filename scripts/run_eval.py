from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_reply_qwen.client import EndpointConfig, OpenAICompatibleClient  # noqa: E402
from im_reply_qwen.evaluation import evaluate_reply  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_model(client: OpenAICompatibleClient, cases: list[dict]) -> dict:
    records: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        output = client.generate(case["messages"], temperature=0.0)
        latency = time.perf_counter() - started
        constraints = case["constraints"]
        scores = evaluate_reply(
            output,
            max_chars=constraints["max_chars"],
            required_all=constraints.get("required_all", []),
            required_any=constraints.get("required_any", []),
            forbidden_any=constraints.get("forbidden_any", []),
            available_facts=case.get("available_facts", ""),
        )
        records.append(
            {
                "id": case["id"],
                "scenario": case["scenario"],
                "reference": case["reference"],
                "output": output,
                "latency_seconds": round(latency, 4),
                "scores": scores.to_dict(),
            }
        )

    total = len(records)
    summary = {
        "cases": total,
        "all_rules_pass_rate": sum(r["scores"]["all_rules_pass"] for r in records) / total,
        "length_pass_rate": sum(r["scores"]["length_pass"] for r in records) / total,
        "required_all_pass_rate": sum(r["scores"]["required_all_pass"] for r in records)
        / total,
        "required_pass_rate": sum(r["scores"]["required_pass"] for r in records) / total,
        "forbidden_pass_rate": sum(r["scores"]["forbidden_pass"] for r in records) / total,
        "no_think_pass_rate": sum(r["scores"]["no_think_pass"] for r in records) / total,
        "mean_latency_seconds": sum(r["latency_seconds"] for r in records) / total,
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base and LoRA model with fixed cases")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tuned-model", default="im-reply-lora")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "eval.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "eval.json")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results = {}
    for label, model in (("base", args.base_model), ("lora", args.tuned_model)):
        print(f"Evaluating {label}: {model}")
        client = OpenAICompatibleClient(
            EndpointConfig(base_url=args.base_url, model=model, api_key=args.api_key)
        )
        results[label] = evaluate_model(client, cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in results.items()}, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
