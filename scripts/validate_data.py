from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_reply_qwen.evaluation import evaluate_reply  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def validate_training(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_users: set[str] = set()
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            errors.append(f"row {index}: training messages must contain system/user/assistant")
            continue
        roles = [message.get("role") for message in messages]
        if roles[-1] != "assistant" or "user" not in roles:
            errors.append(f"row {index}: invalid role order {roles}")
        user_text = "\n".join(
            message.get("content", "") for message in messages if message.get("role") == "user"
        )
        if user_text in seen_users:
            errors.append(f"row {index}: duplicate user prompt")
        seen_users.add(user_text)
        if "/no_think" not in user_text:
            errors.append(f"row {index}: missing /no_think")
    return errors


def validate_eval(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        case_id = row.get("id")
        if not case_id or case_id in seen_ids:
            errors.append(f"row {index}: missing or duplicate id")
        seen_ids.add(case_id)
        if not row.get("messages") or not row.get("reference"):
            errors.append(f"row {index}: messages and reference are required")
        constraints = row.get("constraints", {})
        if not isinstance(constraints.get("max_chars"), int):
            errors.append(f"row {index}: constraints.max_chars must be an integer")
            continue
        reference_scores = evaluate_reply(
            row["reference"],
            max_chars=constraints["max_chars"],
            required_all=constraints.get("required_all", []),
            required_any=constraints.get("required_any", []),
            forbidden_any=constraints.get("forbidden_any", []),
            available_facts=row.get("available_facts", ""),
        )
        if not reference_scores.all_rules_pass:
            errors.append(
                f"row {index}: reference violates its own constraints: "
                f"{reference_scores.to_dict()}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=["train", "eval"], required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.path)
    errors = validate_training(rows) if args.kind == "train" else validate_eval(rows)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: {args.path} contains {len(rows)} valid {args.kind} rows")


if __name__ == "__main__":
    main()
