from __future__ import annotations

import os
import sys
import time
import importlib
import inspect
from contextlib import nullcontext
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from im_reply_qwen.client import (  # noqa: E402
    EndpointConfig,
    OpenAICompatibleClient,
    demo_baseline,
)
from im_reply_qwen import evaluation as evaluation_module  # noqa: E402
from im_reply_qwen import prompting as prompting_module  # noqa: E402


# Streamlit reruns app.py without necessarily reloading imported local modules.
# Reload only when a running development server still holds the pre-facts API.
if "available_facts" not in prompting_module.ReplyRequest.__dataclass_fields__:
    prompting_module = importlib.reload(prompting_module)
if "available_facts" not in inspect.signature(evaluation_module.evaluate_reply).parameters:
    evaluation_module = importlib.reload(evaluation_module)

ReplyRequest = prompting_module.ReplyRequest
build_messages = prompting_module.build_messages
evaluate_reply = evaluation_module.evaluate_reply
visible_text = evaluation_module.visible_text


EXAMPLES = {
    "招聘沟通": {
        "history": "HR：方便发一份简历过来吗？\n候选人：已经通过附件发送了。\nHR：收到，您投递的是大模型岗位吗？",
        "goal": "确认投递岗位，并简要说明匹配经历",
        "facts": (
            "使用 Python、Streamlit 开发过 AI 求职助手\n"
            "实现了 PDF、DOCX 简历解析\n"
            "调用过大模型 API\n"
            "做过 Prompt 设计和结果状态管理\n"
            "正在学习 Qwen3 LoRA 微调\n"
            "没有 MNN-Chat 项目经验"
        ),
        "required_any": ["Python", "Streamlit", "简历解析", "大模型", "Prompt", "LoRA", "MNN-Chat"],
        "tone": "自然、积极",
        "max_chars": 70,
    },
    "客服咨询": {
        "history": "用户：订单显示已签收，但我没有收到。\n客服：请问方便提供一下订单号吗？\n用户：订单号是 A1024。",
        "goal": "确认收到订单号并说明下一步核查",
        "facts": "订单号为 A1024\n用户表示订单未收到",
        "required_any": ["A1024", "核查", "查询", "处理"],
        "tone": "专业、安抚",
        "max_chars": 60,
    },
    "日程确认": {
        "history": "同事：我们周三下午讨论一下接口方案？\n我：可以，几点方便？\n同事：三点怎么样？",
        "goal": "确认周三下午三点参加讨论",
        "facts": "已同意周三讨论\n对方建议下午三点",
        "required_any": ["周三", "三点", "15:00"],
        "tone": "简洁、自然",
        "max_chars": 35,
    },
}


def find_default_model_path() -> str:
    configured = os.getenv("LOCAL_MODEL_PATH")
    if configured:
        return configured
    candidate = (
        WORKSPACE_ROOT
        / "work"
        / "modelscope-cache"
        / "models"
        / "Qwen--Qwen3-0.6B"
        / "snapshots"
        / "master"
    )
    return str(candidate) if candidate.exists() else ""


def find_default_adapter_path() -> str:
    configured = os.getenv("LOCAL_ADAPTER_PATH")
    if configured:
        return configured
    checkpoint_root = WORKSPACE_ROOT / "work" / "checkpoints"
    candidates = list(checkpoint_root.glob("**/checkpoint-*"))
    candidates = [path for path in candidates if (path / "adapter_model.safetensors").exists()]
    if not candidates:
        return ""
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def call_api_model(
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
) -> tuple[str, float]:
    started = time.perf_counter()
    client = OpenAICompatibleClient(
        EndpointConfig(
            base_url=base_url,
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )
    )
    output = client.generate(messages)
    return output, time.perf_counter() - started


@st.cache_resource(show_spinner=False)
def load_local_model(model_path: str, adapter_path: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(model_path).exists():
        raise FileNotFoundError(f"找不到基础模型目录：{model_path}")
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"找不到 LoRA checkpoint：{adapter_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境没有识别到 CUDA 显卡")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_local(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    use_adapter: bool,
    max_new_tokens: int = 128,
) -> tuple[str, float]:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    adapter_context = nullcontext() if use_adapter else model.disable_adapter()

    started = time.perf_counter()
    with adapter_context, torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return visible_text(output), elapsed


st.set_page_config(page_title="IM-Reply-Qwen", page_icon="💬", layout="wide")
st.title("IM-Reply-Qwen")
st.caption("中文 IM 可控回复：原始 Qwen3 与 LoRA 适配器并排比较")

with st.sidebar:
    st.header("运行模式")
    run_mode = st.radio(
        "选择推理方式",
        ["离线界面演示", "本地真实模型", "OpenAI-compatible 服务"],
        index=0,
    )

    local_model_path = ""
    local_adapter_path = ""
    base_url = ""
    api_base_model = ""
    api_tuned_model = ""

    if run_mode == "本地真实模型":
        local_model_path = st.text_input("Qwen3 基础模型目录", value=find_default_model_path())
        local_adapter_path = st.text_input(
            "LoRA checkpoint 目录",
            value=find_default_adapter_path(),
            key="local_adapter_path_profile_v1",
        )
        model_ready = Path(local_model_path).exists() if local_model_path else False
        adapter_ready = Path(local_adapter_path).exists() if local_adapter_path else False
        if model_ready and adapter_ready:
            st.success("已找到基础模型和 LoRA checkpoint")
        else:
            st.warning("请检查基础模型和 checkpoint 路径")
        if st.button("清除模型缓存并重新加载"):
            load_local_model.clear()
            st.success("缓存已清除，下次生成时会重新加载")

    elif run_mode == "OpenAI-compatible 服务":
        base_url = st.text_input(
            "OpenAI-compatible URL",
            value=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        )
        api_base_model = st.text_input(
            "原始模型名",
            value=os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B"),
        )
        api_tuned_model = st.text_input(
            "LoRA 模型名",
            value=os.getenv("TUNED_MODEL", "im-reply-lora"),
        )
    else:
        st.info("只验证页面交互，不代表真实模型效果")

selected = st.selectbox("示例场景", options=list(EXAMPLES))
example = EXAMPLES[selected]
scenario = st.text_input("场景", value=selected)
history = st.text_area("历史对话", value=example["history"], height=180)
available_facts = st.text_area(
    "可用事实（模型只能使用这些信息）",
    value=example["facts"],
    height=150,
    help="每行一条。公开 Demo 请勿填写手机号、邮箱、真实姓名等隐私信息。",
)
goal = st.text_input("回复目标", value=example["goal"])
tone = st.text_input("语气", value=example["tone"])
max_chars = st.slider("最大字符数", min_value=10, max_value=160, value=example["max_chars"])
st.caption(f"目标完成检查：回复至少包含一项关键信息——{'、'.join(example['required_any'])}")

if st.button("生成并比较", type="primary", use_container_width=True):
    request = ReplyRequest(
        scenario=scenario,
        history=history,
        goal=goal,
        available_facts=available_facts,
        tone=tone,
        max_chars=max_chars,
    )
    messages = build_messages(request)

    if run_mode == "离线界面演示":
        base_output = demo_baseline(goal, max_chars)
        tuned_output = "离线演示模式未连接 LoRA 模型。请选择“本地真实模型”查看真实结果。"
        base_latency = tuned_latency = 0.0

    elif run_mode == "本地真实模型":
        try:
            with st.spinner("首次加载 Qwen3 和 LoRA，通常需要数秒……"):
                tokenizer, local_model = load_local_model(local_model_path, local_adapter_path)
            base_output, base_latency = generate_local(
                local_model,
                tokenizer,
                messages,
                use_adapter=False,
            )
            tuned_output, tuned_latency = generate_local(
                local_model,
                tokenizer,
                messages,
                use_adapter=True,
            )
        except Exception as exc:
            st.error(f"本地模型加载或推理失败：{exc}")
            st.stop()

    else:
        try:
            base_output, base_latency = call_api_model(base_url, api_base_model, messages)
            tuned_output, tuned_latency = call_api_model(base_url, api_tuned_model, messages)
        except Exception as exc:
            st.error(f"推理服务调用失败：{exc}")
            st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("原始模型")
        st.write(base_output)
        base_scores = evaluate_reply(
            base_output,
            max_chars=max_chars,
            required_any=example["required_any"],
            available_facts=available_facts,
        )
        st.json({**base_scores.to_dict(), "latency_seconds": round(base_latency, 3)})
    with right:
        st.subheader("LoRA 模型")
        st.write(tuned_output)
        tuned_scores = evaluate_reply(
            tuned_output,
            max_chars=max_chars,
            required_any=example["required_any"],
            available_facts=available_facts,
        )
        st.json({**tuned_scores.to_dict(), "latency_seconds": round(tuned_latency, 3)})

    with st.expander("查看发送给模型的 messages"):
        st.json(messages)
