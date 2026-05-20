import unittest

from app.rag_service import distance_to_relevance_score, filter_chunks_by_relevance


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
