from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = (
    "你是中文 IM 回复助手。请根据历史对话和回复目标生成一条可以直接发送的回复。"
    "必须保持人物、时间、数字和承诺等事实一致；信息不足时应澄清，不得自行编造；"
    "遵守给定语气和长度限制；只输出回复正文，不解释思考过程。"
)


@dataclass(frozen=True)
class ReplyRequest:
    scenario: str
    history: str
    goal: str
    available_facts: str = ""
    tone: str = "自然、礼貌"
    max_chars: int = 60

    def validate(self) -> None:
        if not self.scenario.strip():
            raise ValueError("scenario must not be empty")
        if not self.history.strip():
            raise ValueError("history must not be empty")
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if not 10 <= self.max_chars <= 300:
            raise ValueError("max_chars must be between 10 and 300")


def build_user_prompt(request: ReplyRequest) -> str:
    request.validate()
    facts = request.available_facts.strip() or "未提供可用事实；不得自行补充个人经历或能力。"
    return (
        f"【场景】{request.scenario.strip()}\n"
        f"【历史对话】\n{request.history.strip()}\n"
        f"【可用事实】\n{facts}\n"
        "【事实边界】只能使用历史对话和可用事实中的信息；"
        "不得把“正在学习”改写成“熟练掌握”，也不得把“没有经验”改写成“做过项目”。\n"
        f"【回复目标】{request.goal.strip()}\n"
        f"【语气】{request.tone.strip()}\n"
        f"【长度限制】不超过 {request.max_chars} 个字符\n"
        "/no_think"
    )


def build_messages(request: ReplyRequest) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(request)},
    ]
