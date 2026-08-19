from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_reply_qwen.evaluation import visible_text  # noqa: E402
from im_reply_qwen.prompting import ReplyRequest, build_messages  # noqa: E402


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
    return visible_text(tokenizer.decode(new_tokens, skip_special_tokens=True)), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the base model and a local LoRA checkpoint")
    parser.add_argument("--model", required=True, help="Local base model directory or hub model ID")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--scenario", default="招聘沟通")
    parser.add_argument(
        "--history",
        default="HR：您有 MNN-Chat 的项目吗？\n候选人：还没有，正在学习。",
    )
    parser.add_argument("--goal", default="如实说明暂无项目，并说明正在制作端侧 Demo")
    parser.add_argument(
        "--facts",
        default="正在学习 Qwen3 LoRA 微调\n没有 MNN-Chat 项目经验",
    )
    parser.add_argument("--tone", default="坦诚、积极")
    parser.add_argument("--max-chars", type=int, default=55)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this comparison is intended for local GPU smoke tests")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    request = ReplyRequest(
        scenario=args.scenario,
        history=args.history,
        goal=args.goal,
        available_facts=args.facts,
        tone=args.tone,
        max_chars=args.max_chars,
    )
    messages = build_messages(request)

    base_output, base_latency = generate(model, tokenizer, messages, args.max_new_tokens)
    tuned_model = PeftModel.from_pretrained(model, str(args.adapter))
    tuned_model.eval()
    tuned_output, tuned_latency = generate(tuned_model, tokenizer, messages, args.max_new_tokens)

    print(
        json.dumps(
            {
                "request": request.__dict__,
                "base": {"output": base_output, "latency_seconds": round(base_latency, 3)},
                "lora": {"output": tuned_output, "latency_seconds": round(tuned_latency, 3)},
                "notice": "Smoke checkpoints are not quality evidence; run the fixed evaluation set.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
