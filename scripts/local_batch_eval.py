from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_reply_qwen.evaluation import evaluate_reply, visible_text  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def generate(model, tokenizer, messages: list[dict], *, adapter: str | None) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    context = model.disable_adapter() if adapter is None else nullcontext()
    if adapter is not None:
        model.set_adapter(adapter)
    started = time.perf_counter()
    with context, torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
    return visible_text(tokenizer.decode(new_tokens, skip_special_tokens=True)), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local base and LoRA adapters")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", action="append", required=True, help="label=checkpoint_path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    adapters: list[tuple[str, str]] = []
    for value in args.adapter:
        label, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--adapter must use label=checkpoint_path")
        adapters.append((label, path))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base_model, adapters[0][1], adapter_name=adapters[0][0])
    for label, path in adapters[1:]:
        model.load_adapter(path, adapter_name=label)
    model.eval()

    cases = load_jsonl(args.data)
    candidates: list[tuple[str, str | None]] = [("base", None)] + [
        (label, label) for label, _ in adapters
    ]
    result: dict[str, dict] = {}
    for label, adapter_name in candidates:
        rows = []
        for case in cases:
            output, latency = generate(model, tokenizer, case["messages"], adapter=adapter_name)
            constraints = case["constraints"]
            scores = evaluate_reply(
                output,
                max_chars=constraints["max_chars"],
                required_all=constraints.get("required_all", []),
                required_any=constraints.get("required_any", []),
                forbidden_any=constraints.get("forbidden_any", []),
                available_facts=case.get("available_facts", ""),
            )
            rows.append(
                {
                    "id": case["id"],
                    "output": output,
                    "scores": scores.to_dict(),
                    "latency_seconds": round(latency, 3),
                }
            )
        total = len(rows)
        result[label] = {
            "summary": {
                "cases": total,
                "all_rules_pass_rate": round(sum(row["scores"]["all_rules_pass"] for row in rows) / total, 4),
                "required_pass_rate": round(sum(row["scores"]["required_pass"] for row in rows) / total, 4),
                "fact_consistency_rate": round(sum(row["scores"]["fact_consistency_pass"] for row in rows) / total, 4),
                "mean_latency_seconds": round(statistics.mean(row["latency_seconds"] for row in rows), 3),
            },
            "rows": rows,
        }

    payload = {"data": str(args.data), "results": result}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({label: value["summary"] for label, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
