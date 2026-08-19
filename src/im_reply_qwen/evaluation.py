from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def visible_text(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


def detect_fact_conflicts(text: str, available_facts: str) -> tuple[str, ...]:
    """Detect obvious positive claims that contradict an explicit negative fact.

    This is deliberately conservative and only checks facts beginning with words
    such as ``没有`` or ``暂无``. It is a guardrail, not a semantic judge.
    """
    normalized_reply = re.sub(r"\s+", "", visible_text(text)).lower()
    conflicts: list[str] = []
    negative_prefixes = ("没有", "暂无", "没做过", "未参与过", "未做过")
    positive_markers = ("做过", "参与过", "负责过", "具备", "拥有", "相关经验", "相关经历", "项目经验")

    for raw_line in available_facts.splitlines():
        line = raw_line.strip().lstrip("-•* ").rstrip("。；;")
        prefix = next((item for item in negative_prefixes if line.startswith(item)), None)
        if not prefix:
            continue
        subject = line[len(prefix) :].strip()
        topic = re.sub(r"(?:相关)?项目经验$|(?:相关)?经验$", "", subject).strip()
        normalized_topic = re.sub(r"\s+", "", topic).lower()
        if not normalized_topic or normalized_topic not in normalized_reply:
            continue

        negative_forms = tuple(
            re.sub(r"\s+", "", f"{item}{topic}").lower() for item in negative_prefixes
        )
        states_negative = any(form in normalized_reply for form in negative_forms)
        makes_positive_claim = any(marker in normalized_reply for marker in positive_markers)
        if makes_positive_claim and not states_negative:
            conflicts.append(line)
    return tuple(conflicts)


@dataclass(frozen=True)
class RuleScores:
    char_count: int
    length_pass: bool
    required_all_pass: bool
    required_pass: bool
    forbidden_pass: bool
    no_think_pass: bool
    fact_consistency_pass: bool
    fact_conflicts: tuple[str, ...]
    all_rules_pass: bool

    def to_dict(self) -> dict[str, int | bool | tuple[str, ...]]:
        return asdict(self)


def evaluate_reply(
    text: str,
    *,
    max_chars: int,
    required_all: Iterable[str] = (),
    required_any: Iterable[str] = (),
    forbidden_any: Iterable[str] = (),
    available_facts: str = "",
) -> RuleScores:
    raw = text.strip()
    visible = visible_text(raw)
    required_all_items = [item for item in required_all if item]
    required_any_items = [item for item in required_any if item]
    forbidden = [item for item in forbidden_any if item]

    length_pass = len(visible) <= max_chars
    required_all_pass = all(item in visible for item in required_all_items)
    required_pass = not required_any_items or any(item in visible for item in required_any_items)
    forbidden_pass = not any(item in visible for item in forbidden)
    no_think_pass = "<think>" not in raw.lower()
    fact_conflicts = detect_fact_conflicts(visible, available_facts)
    fact_consistency_pass = not fact_conflicts
    all_rules_pass = (
        length_pass
        and required_all_pass
        and required_pass
        and forbidden_pass
        and no_think_pass
        and fact_consistency_pass
    )
    return RuleScores(
        char_count=len(visible),
        length_pass=length_pass,
        required_all_pass=required_all_pass,
        required_pass=required_pass,
        forbidden_pass=forbidden_pass,
        no_think_pass=no_think_pass,
        fact_consistency_pass=fact_consistency_pass,
        fact_conflicts=fact_conflicts,
        all_rules_pass=all_rules_pass,
    )
