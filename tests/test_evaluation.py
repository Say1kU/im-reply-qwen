import unittest

from im_reply_qwen.evaluation import detect_fact_conflicts, evaluate_reply, visible_text


class EvaluationTests(unittest.TestCase):
    def test_rule_scores(self) -> None:
        scores = evaluate_reply(
            "可以，周三下午三点见。",
            max_chars=20,
            required_all=["周三", "三点"],
            forbidden_any=["周四"],
        )
        self.assertTrue(scores.all_rules_pass)

    def test_think_is_not_public_output(self) -> None:
        text = "<think>需要简短回复</think>好的，收到。"
        self.assertEqual(visible_text(text), "好的，收到。")
        scores = evaluate_reply(text, max_chars=20)
        self.assertFalse(scores.no_think_pass)
        self.assertFalse(scores.all_rules_pass)

    def test_detects_claim_contradicting_negative_fact(self) -> None:
        facts = "正在学习 Qwen3 LoRA 微调\n没有 MNN-Chat 项目经验"
        conflicts = detect_fact_conflicts("我做过 MNN-Chat 项目，具备相关经验。", facts)
        self.assertEqual(conflicts, ("没有 MNN-Chat 项目经验",))
        scores = evaluate_reply(
            "我做过 MNN-Chat 项目，具备相关经验。",
            max_chars=50,
            available_facts=facts,
        )
        self.assertFalse(scores.fact_consistency_pass)
        self.assertFalse(scores.all_rules_pass)

    def test_accepts_honest_negative_fact(self) -> None:
        facts = "没有 MNN-Chat 项目经验"
        scores = evaluate_reply(
            "目前没有 MNN-Chat 项目经验，正在学习相关技术。",
            max_chars=50,
            available_facts=facts,
        )
        self.assertTrue(scores.fact_consistency_pass)


if __name__ == "__main__":
    unittest.main()
