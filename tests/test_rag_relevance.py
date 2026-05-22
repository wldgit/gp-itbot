import unittest

from unittest.mock import patch

from app.rag_service import (
    append_sources_to_answer,
    chunk_text_preview,
    dedupe_sources,
    distance_to_relevance_score,
    evaluate_retrieval_quality,
    filter_chunks_by_relevance,
    parse_groundedness_response,
)


class EvaluateRetrievalQualityTests(unittest.TestCase):
    @patch("app.rag_service.settings")
    def test_no_chunks_after_min_filter(self, mock_settings):
        mock_settings.rag_confident_score = 0.55
        quality, best = evaluate_retrieval_quality([])
        self.assertEqual(quality, "no_context")
        self.assertEqual(best, 0.0)

    @patch("app.rag_service.settings")
    def test_confident(self, mock_settings):
        mock_settings.rag_confident_score = 0.55
        items = [{"relevance_score": 0.6}, {"relevance_score": 0.48}]
        quality, best = evaluate_retrieval_quality(items)
        self.assertEqual(quality, "confident")
        self.assertAlmostEqual(best, 0.6)

    @patch("app.rag_service.settings")
    def test_borderline_between_min_and_confident(self, mock_settings):
        mock_settings.rag_confident_score = 0.55
        items = [{"relevance_score": 0.47}, {"relevance_score": 0.36}]
        quality, best = evaluate_retrieval_quality(items)
        self.assertEqual(quality, "borderline")
        self.assertAlmostEqual(best, 0.47)


class ParseGroundednessResponseTests(unittest.TestCase):
    def test_direct_answer_true(self):
        has_direct, reason = parse_groundedness_response(
            '{"has_direct_answer": true, "reason": "есть шаги BitLocker"}'
        )
        self.assertTrue(has_direct)
        self.assertIn("BitLocker", reason)

    def test_direct_answer_false(self):
        has_direct, _ = parse_groundedness_response(
            '{"has_direct_answer": false, "reason": "только блокировка аккаунта"}'
        )
        self.assertFalse(has_direct)


class DedupeSourcesTests(unittest.TestCase):
    def test_removes_duplicates_preserves_order(self):
        self.assertEqual(
            dedupe_sources(["vpn.md", "faq.txt", "vpn.md"]),
            ["vpn.md", "faq.txt"],
        )

    def test_skips_empty_values(self):
        self.assertEqual(dedupe_sources(["a.md", "", "  ", "a.md"]), ["a.md"])


class AppendSourcesToAnswerTests(unittest.TestCase):
    def test_empty_sources_returns_answer_unchanged(self):
        self.assertEqual(append_sources_to_answer("Ответ.", []), "Ответ.")

    def test_appends_sources_block(self):
        result = append_sources_to_answer(
            "Шаги выполнены.",
            ["vpn_fortinet_setup.md", "FAQ_connections.txt"],
        )
        self.assertIn("Шаги выполнены.", result)
        self.assertIn("Использованные документы:", result)
        self.assertIn("- vpn_fortinet_setup.md", result)
        self.assertIn("- FAQ_connections.txt", result)

    def test_dedupes_sources_in_footer(self):
        result = append_sources_to_answer("OK", ["vpn.md", "faq.txt", "vpn.md"])
        self.assertEqual(result.count("- vpn.md"), 1)


class ChunkTextPreviewTests(unittest.TestCase):
    def test_collapses_newlines(self):
        preview = chunk_text_preview("line one\nline two", max_chars=120)
        self.assertEqual(preview, "line one line two")

    def test_truncates_long_text(self):
        preview = chunk_text_preview("a" * 200, max_chars=120)
        self.assertEqual(len(preview), 123)
        self.assertTrue(preview.endswith("..."))


class DistanceToRelevanceScoreTests(unittest.TestCase):
    def test_none_returns_zero(self):
        self.assertEqual(distance_to_relevance_score(None), 0.0)

    def test_zero_distance(self):
        self.assertEqual(distance_to_relevance_score(0), 1.0)

    def test_distance_one(self):
        self.assertAlmostEqual(distance_to_relevance_score(1), 0.5)

    def test_distance_two(self):
        self.assertAlmostEqual(distance_to_relevance_score(2), 1.0 / 3.0)

    def test_negative_distance_treated_as_zero(self):
        self.assertEqual(distance_to_relevance_score(-1), 1.0)


class FilterChunksByRelevanceTests(unittest.TestCase):
    def _filter(self, distances: list[float | None], threshold: float = 0.5):
        docs = [f"doc-{idx}" for idx in range(len(distances))]
        metadatas = [{"source": f"file-{idx}.md", "section": f"sec-{idx}"} for idx in range(len(distances))]
        return filter_chunks_by_relevance(docs, metadatas, distances, threshold=threshold)

    def test_distance_one_accepted_at_threshold_half(self):
        accepted, rejected = self._filter([1.0], threshold=0.5)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)

    def test_distance_two_rejected_at_threshold_half(self):
        accepted, rejected = self._filter([2.0], threshold=0.5)
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)

    def test_sorted_by_relevance_desc(self):
        accepted, _ = self._filter([2.0, 0.0, 1.0], threshold=0.3)
        scores = [item["relevance_score"] for item in accepted]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
