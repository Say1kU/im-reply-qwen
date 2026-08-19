from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_reply_qwen.evaluation import evaluate_reply  # noqa: E402
from im_reply_qwen.prompting import ReplyRequest, build_messages  # noqa: E402


PROFILE_FACTS = """使用 Python、Streamlit 开发过 AI 求职助手
实现了 PDF、DOCX 简历解析
调用过大模型 API
做过 Prompt 设计和结果状态管理
正在学习 Qwen3 LoRA 微调
没有 MNN-Chat 项目经验"""

FACT_CASES = [
    ("Python", "使用 Python 开发过 AI 求职助手", "我使用 Python 开发过 AI 求职助手。"),
    ("Streamlit", "使用 Streamlit 做过应用界面", "我使用 Streamlit 完成过 AI 求职助手界面。"),
    ("简历解析", "实现 PDF、DOCX 简历解析", "我实现过 PDF、DOCX 简历解析功能。"),
    ("大模型 API", "调用大模型 API", "我有大模型 API 调用和应用接入经验。"),
    ("Prompt", "进行 Prompt 设计", "我做过 Prompt 设计，并处理过结果状态管理。"),
    ("状态管理", "处理结果状态管理", "我实现过大模型结果的状态管理。"),
    ("LoRA", "学习 Qwen3 LoRA 微调", "我正在学习 Qwen3 LoRA 微调，并已跑通训练流程。"),
    ("MNN-Chat", "如实说明 MNN-Chat 经验", "我目前没有 MNN-Chat 项目经验，正在学习相关技术。"),
]

HR_QUESTIONS = [
    "HR：可以简单介绍一项与岗位相关的经历吗？",
    "招聘方：你的经历与这个岗位有哪些匹配点？",
    "HR：方便补充一下相关技术经验吗？",
    "面试官：你有哪些可以落地展示的能力？",
    "招聘方：请用一句话说明你的相关项目经验。",
    "HR：你在大模型应用方面做过什么？",
    "面试官：这项能力有实际项目支撑吗？",
    "HR：能否简要说明你的技术匹配度？",
]

TONES = ["自然、积极", "坦诚、简洁", "专业、礼貌", "自信但不夸大"]


def training_row(history: str, goal: str, answer: str, index: int) -> dict:
    request = ReplyRequest(
        scenario="招聘沟通",
        history=history,
        goal=goal,
        available_facts=PROFILE_FACTS,
        tone=TONES[index % len(TONES)],
        max_chars=90,
    )
    return {"messages": [*build_messages(request), {"role": "assistant", "content": answer}]}


def build_training_rows() -> list[dict]:
    rows: list[dict] = []
    for fact_index, (_, goal, answer) in enumerate(FACT_CASES):
        for question_index, question in enumerate(HR_QUESTIONS):
            prefix = ["您好，", "可以，", "当然，", ""][question_index % 4]
            rows.append(
                training_row(
                    question,
                    goal,
                    f"{prefix}{answer}",
                    fact_index + question_index,
                )
            )

    combined = [
        (
            "HR：你做过哪些大模型应用项目？",
            "结合真实项目说明两项匹配经历",
            "我用 Python、Streamlit 开发过 AI 求职助手，也实现了 PDF、DOCX 简历解析。",
        ),
        (
            "HR：你有微调和端侧部署经验吗？",
            "区分已经完成和正在学习的内容",
            "我正在学习 Qwen3 LoRA 微调；目前没有 MNN-Chat 项目经验，不会夸大这部分经历。",
        ),
        (
            "HR：你的 AI 求职助手包含哪些功能？",
            "介绍已经实现的功能",
            "项目支持 PDF、DOCX 简历解析、大模型 API 调用以及结果状态管理。",
        ),
        (
            "HR：你熟练掌握 Qwen3 微调了吗？",
            "如实说明仍在学习",
            "目前还不能说熟练掌握，我正在学习 Qwen3 LoRA 微调并持续实践。",
        ),
        (
            "HR：你做过 MNN-Chat 吗？",
            "明确说明没有相关项目经验",
            "目前没有 MNN-Chat 项目经验，我正在学习端侧推理相关技术。",
        ),
        (
            "HR：可以发一下你的微调项目吗？",
            "说明项目仍在完善，不虚构公开链接",
            "项目正在完善和评测中，整理好可公开版本后我会及时分享。",
        ),
        (
            "HR：你主要做模型研究还是应用开发？",
            "说明当前经验侧重",
            "我目前更侧重大模型应用开发，做过 API 接入、Prompt 设计和 Streamlit 应用。",
        ),
        (
            "HR：你能独立完成 Demo 吗？",
            "用已完成经历回答",
            "可以，我使用 Python 和 Streamlit 独立开发过 AI 求职助手 Demo。",
        ),
    ]
    for repeat in range(4):
        for case_index, (history, goal, answer) in enumerate(combined):
            suffix = ["", "请简短回答。", "请不要夸大经验。", "请突出可验证内容。"][repeat]
            rows.append(training_row(f"{history}\nHR：{suffix}" if suffix else history, goal, answer, repeat + case_index))
    return rows


def eval_row(
    case_id: str,
    history: str,
    goal: str,
    reference: str,
    *,
    required_any: list[str],
    forbidden_any: list[str],
) -> dict:
    request = ReplyRequest(
        scenario="招聘沟通",
        history=history,
        goal=goal,
        available_facts=PROFILE_FACTS,
        tone="自然、坦诚",
        max_chars=90,
    )
    constraints = {
        "max_chars": 90,
        "required_any": required_any,
        "forbidden_any": forbidden_any,
    }
    scores = evaluate_reply(
        reference,
        max_chars=90,
        required_any=required_any,
        forbidden_any=forbidden_any,
        available_facts=PROFILE_FACTS,
    )
    if not scores.all_rules_pass:
        raise ValueError(f"invalid reference {case_id}: {scores.to_dict()}")
    return {
        "id": case_id,
        "scenario": "招聘沟通",
        "messages": build_messages(request),
        "available_facts": PROFILE_FACTS,
        "reference": reference,
        "constraints": constraints,
    }


def build_eval_rows() -> list[dict]:
    cases = [
        ("profile-001", "HR：你做过什么 AI 应用？", "说明一项已完成项目", "我使用 Python、Streamlit 开发过 AI 求职助手。", ["Python", "Streamlit", "求职助手"], ["MNN-Chat 项目"]),
        ("profile-002", "HR：你有 MNN-Chat 项目经验吗？", "如实说明经验边界", "目前没有 MNN-Chat 项目经验，我正在学习相关技术。", ["没有", "暂无", "还没有"], ["做过 MNN-Chat", "熟练掌握 MNN-Chat"]),
        ("profile-003", "HR：你熟练掌握 LoRA 微调吗？", "避免把学习描述成熟练", "我正在学习 Qwen3 LoRA 微调，目前仍在积累实践经验。", ["正在学习", "学习"], ["熟练掌握", "精通"]),
        ("profile-004", "HR：简历解析支持什么格式？", "说明已经支持的格式", "我实现的简历解析功能支持 PDF 和 DOCX。", ["PDF", "DOCX"], ["图片", "Excel"]),
        ("profile-005", "HR：你做过大模型接口接入吗？", "确认真实 API 调用经验", "做过，我在 AI 求职助手中接入并调用过大模型 API。", ["API", "大模型"], ["MNN-Chat"]),
        ("profile-006", "HR：Prompt 方面有哪些经验？", "说明真实 Prompt 工作", "我做过 Prompt 设计，也处理过模型结果的状态管理。", ["Prompt", "状态管理"], ["精通"]),
        ("profile-007", "HR：你有端侧大模型落地项目吗？", "坦诚说明尚无 MNN 项目", "目前没有 MNN-Chat 项目经验，正在学习端侧推理。", ["没有", "暂无", "正在学习"], ["已经落地", "量产"]),
        ("profile-008", "HR：能给一个公开的项目链接吗？", "项目未公开时不要编造链接", "项目仍在整理，完成公开版本后我会及时分享。", ["整理", "完成", "公开"], ["http://", "https://"]),
        ("profile-009", "HR：你主要使用什么语言开发？", "说明真实开发语言", "我主要使用 Python 开发 AI 求职助手。", ["Python"], ["Java", "C++"]),
        ("profile-010", "HR：这个项目的界面用什么做的？", "说明真实界面技术", "AI 求职助手的界面使用 Streamlit 开发。", ["Streamlit"], ["Django", "React"]),
        ("profile-011", "HR：你能处理 Word 简历吗？", "说明 DOCX 解析能力", "可以，我实现过 DOCX 简历解析，也支持 PDF。", ["DOCX", "PDF"], ["Excel"]),
        ("profile-012", "HR：你已经完成 Qwen3 微调项目了吗？", "准确说明目前进展", "我正在学习并实践 Qwen3 LoRA 微调，项目还在完善。", ["学习", "实践", "完善"], ["已经完成", "成熟上线"]),
        ("profile-013", "HR：你做过哪些结果处理？", "说明结果状态管理经验", "我做过大模型结果状态管理，处理生成过程和结果展示。", ["状态管理", "结果"], ["分布式训练"]),
        ("profile-014", "HR：你更偏应用还是算法？", "准确描述当前方向", "我目前更偏大模型应用开发，也在补充 Qwen3 LoRA 微调能力。", ["应用开发", "LoRA"], ["算法专家"]),
        ("profile-015", "HR：一句话说明你和岗位的匹配点。", "引用至少一项真实经历", "我做过 Python 大模型应用、简历解析和 Prompt 设计，并在学习 LoRA 微调。", ["Python", "简历解析", "Prompt", "LoRA"], ["MNN-Chat 项目经验"]),
        ("profile-016", "HR：你的项目已经部署到生产环境了吗？", "信息不足时不虚构部署状态", "目前可确认已完成应用开发，是否达到生产部署标准还需结合实际环境验证。", ["验证", "确认"], ["已经生产部署", "稳定运行一年"]),
    ]
    return [
        eval_row(case_id, history, goal, reference, required_any=required, forbidden_any=forbidden)
        for case_id, history, goal, reference, required, forbidden in cases
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    train_rows = build_training_rows()
    eval_rows = build_eval_rows()
    write_jsonl(PROJECT_ROOT / "data" / "profile_train.jsonl", train_rows)
    write_jsonl(PROJECT_ROOT / "data" / "profile_eval.jsonl", eval_rows)
    print(f"wrote {len(train_rows)} training rows and {len(eval_rows)} evaluation rows")


if __name__ == "__main__":
    main()
