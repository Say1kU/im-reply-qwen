import unittest

from im_reply_qwen.prompting import ReplyRequest, build_messages, build_user_prompt


class PromptingTests(unittest.TestCase):
    def test_build_prompt_contains_constraints(self) -> None:
        request = ReplyRequest(
            scenario="招聘沟通",
            history="HR：方便沟通吗？",
            goal="确认方便",
            available_facts="使用 Python 开发过项目\n没有 MNN-Chat 项目经验",
            tone="自然",
            max_chars=30,
        )
        prompt = build_user_prompt(request)
        self.assertIn("招聘沟通", prompt)
        self.assertIn("【可用事实】", prompt)
        self.assertIn("没有 MNN-Chat 项目经验", prompt)
        self.assertIn("不得把“没有经验”改写成“做过项目”", prompt)
        self.assertIn("不超过 30 个字符", prompt)
        self.assertTrue(prompt.endswith("/no_think"))
        self.assertEqual(len(build_messages(request)), 2)

    def test_rejects_empty_history(self) -> None:
        request = ReplyRequest(scenario="测试", history="", goal="回复")
        with self.assertRaises(ValueError):
            build_user_prompt(request)


if __name__ == "__main__":
    unittest.main()
