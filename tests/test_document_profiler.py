import unittest

from ingest.document_profiler import parse_profiler_response, validate_document_profile


class DocumentProfilerParseTests(unittest.TestCase):
    def test_valid_profile(self):
        raw = (
            '{"doc_type": "faq", "chunking_strategy": "qa_pairs", '
            '"max_chunk_tokens": 450, "overlap_tokens": 50, "confidence": 0.9, '
            '"signals": ["qa format"], "section_profiles": []}'
        )
        profile = parse_profiler_response(raw)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.doc_type, "faq")
        self.assertEqual(profile.chunking_strategy, "qa_pairs")

    def test_invalid_json_returns_none(self):
        self.assertIsNone(parse_profiler_response("not json"))

    def test_low_confidence_returns_none(self):
        raw = (
            '{"doc_type": "faq", "chunking_strategy": "qa_pairs", '
            '"max_chunk_tokens": 450, "overlap_tokens": 50, "confidence": 0.1, '
            '"signals": [], "section_profiles": []}'
        )
        self.assertIsNone(parse_profiler_response(raw))

    def test_validate_normalizes_faq_strategy(self):
        profile = validate_document_profile(
            {
                "doc_type": "faq",
                "chunking_strategy": "sections",
                "max_chunk_tokens": 450,
                "overlap_tokens": 50,
                "confidence": 0.9,
            }
        )
        self.assertEqual(profile.chunking_strategy, "qa_pairs")

    def test_validate_normalizes_support_contacts_strategy(self):
        profile = validate_document_profile(
            {
                "doc_type": "support_contacts",
                "chunking_strategy": "whole_document",
                "max_chunk_tokens": 500,
                "overlap_tokens": 0,
                "confidence": 0.9,
            }
        )
        self.assertEqual(profile.chunking_strategy, "sections")


if __name__ == "__main__":
    unittest.main()
