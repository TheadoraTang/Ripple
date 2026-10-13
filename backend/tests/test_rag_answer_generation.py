import unittest


class RagAnswerGenerationTest(unittest.TestCase):
    def test_citation_only_answer_is_not_substantive(self):
        from app.services.rag_service import _is_substantive_answer

        self.assertFalse(_is_substantive_answer("[1][2]"))
        self.assertFalse(_is_substantive_answer(" [1] [2] "))

    def test_normal_answer_is_substantive(self):
        from app.services.rag_service import _is_substantive_answer

        self.assertTrue(_is_substantive_answer("Steam 最新消息主要集中在新主机和游戏发布。 [1][2]"))

    def test_fallback_answer_uses_retrieved_sources(self):
        from app.services.rag_service import _fallback_answer

        answer = _fallback_answer(
            "Steam 最新内容",
            [
                {
                    "title": "V社新主机正式公布",
                    "feed_title": "游研社",
                    "snippet": "Valve 公布新一代 Steam Machine 主机，强调硬件和 SteamOS。",
                },
                {
                    "title": "《时之笛》重制传闻",
                    "feed_title": "游研社",
                    "snippet": "报道称重制版可能在 2026 年内发售。",
                },
            ],
        )

        self.assertIn("相关内容可以先概括为", answer)
        self.assertIn("V社新主机正式公布", answer)
        self.assertIn("[1]", answer)
        self.assertIn("《时之笛》重制传闻", answer)
        self.assertIn("[2]", answer)

    def test_clean_generated_answer_removes_dangling_citation_line(self):
        from app.services.rag_service import _clean_generated_answer

        answer = _clean_generated_answer("ChinaJoy 将在上海举办。\n\n2[1][2]")

        self.assertEqual(answer, "ChinaJoy 将在上海举办。")

    def test_broad_question_with_too_few_citations_uses_fallback(self):
        from app.services.rag_service import _should_use_fallback_answer

        sources = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

        self.assertTrue(_should_use_fallback_answer("ChinaJoy 相关内容", "主要是展会信息。[1]", sources))
        self.assertFalse(_should_use_fallback_answer("ChinaJoy 相关内容", "展会信息见 [1]，行业动态见 [2]。", sources))

    def test_prompt_requires_body_not_citations_only(self):
        from app.services.rag_service import _build_prompts

        system_prompt, user_prompt = _build_prompts(
            "Steam 最新内容",
            [{"title": "Steam 新闻", "feed_title": "Feed", "snippet": "Steam Deck 有新消息。"}],
        )

        self.assertIn("不能只输出引用编号", system_prompt)
        self.assertIn("先用 2-4 句直接总结答案", system_prompt)
        self.assertIn("覆盖前 3-5 个相关来源", system_prompt)
        self.assertIn("Steam 最新内容", user_prompt)


if __name__ == "__main__":
    unittest.main()
