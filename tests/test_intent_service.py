import unittest
from unittest.mock import MagicMock, patch

from app.intent_service import (
    DEFAULT_INTENT,
    IntentResult,
    IntentService,
    _call_intent_classifier,
    _is_no_sampling_model,
    _normalize_intent,
    parse_classifier_response,
    resolve_intent_decision,
    resolve_reasoning_effort,
)


class ParseClassifierResponseTests(unittest.TestCase):
    def test_valid_greeting(self):
        result = parse_classifier_response('{"intent": "greeting", "confidence": 0.91}')
        self.assertEqual(result.intent, "greeting")
        self.assertAlmostEqual(result.confidence, 0.91)

    def test_valid_out_of_scope(self):
        result = parse_classifier_response('{"intent": "out_of_scope", "confidence": 0.88}')
        self.assertEqual(result.intent, "out_of_scope")

    def test_valid_support_contact_request(self):
        result = parse_classifier_response(
            '{"intent": "support_contact_request", "confidence": 0.91}'
        )
        self.assertEqual(result.intent, "support_contact_request")

    def test_valid_it_support_request(self):
        result = parse_classifier_response(
            '{"intent": "it_support_request", "confidence": 0.85}'
        )
        self.assertEqual(result.intent, "it_support_request")

    def test_legacy_offtopic_maps_to_out_of_scope(self):
        result = parse_classifier_response('{"intent": "offtopic", "confidence": 0.9}')
        self.assertEqual(result.intent, "out_of_scope")

    def test_legacy_it_question_maps_to_it_support_request(self):
        result = parse_classifier_response('{"intent": "it_question", "confidence": 0.9}')
        self.assertEqual(result.intent, "it_support_request")

    def test_legacy_support_contacts_maps_to_support_contact_request(self):
        result = parse_classifier_response('{"intent": "support_contacts", "confidence": 0.9}')
        self.assertEqual(result.intent, "support_contact_request")

    def test_invalid_json_fallback(self):
        result = parse_classifier_response("not json")
        self.assertEqual(result.intent, DEFAULT_INTENT)
        self.assertEqual(result.confidence, 0.0)

    def test_unknown_intent_fallback(self):
        result = parse_classifier_response('{"intent": "unknown", "confidence": 0.9}')
        self.assertEqual(result.intent, DEFAULT_INTENT)

    def test_json_in_code_fence(self):
        raw = '```json\n{"intent": "out_of_scope", "confidence": 0.85}\n```'
        result = parse_classifier_response(raw)
        self.assertEqual(result.intent, "out_of_scope")


class ResolveIntentDecisionTests(unittest.TestCase):
    @patch("app.intent_service.settings")
    def test_greeting_above_threshold(self, mock_settings):
        mock_settings.intent_confidence_threshold = 0.65
        decision = resolve_intent_decision(IntentResult(intent="greeting", confidence=0.9))
        self.assertEqual(decision, "greeting")

    @patch("app.intent_service.settings")
    def test_greeting_low_confidence_still_no_rag(self, mock_settings):
        mock_settings.intent_confidence_threshold = 0.65
        decision = resolve_intent_decision(IntentResult(intent="greeting", confidence=0.0))
        self.assertEqual(decision, "greeting")

    @patch("app.intent_service.settings")
    def test_out_of_scope_above_threshold(self, mock_settings):
        mock_settings.intent_confidence_threshold = 0.65
        decision = resolve_intent_decision(IntentResult(intent="out_of_scope", confidence=0.85))
        self.assertEqual(decision, "out_of_scope")

    @patch("app.intent_service.settings")
    def test_it_support_request_goes_to_rag(self, mock_settings):
        mock_settings.intent_confidence_threshold = 0.65
        decision = resolve_intent_decision(
            IntentResult(intent="it_support_request", confidence=0.99)
        )
        self.assertEqual(decision, "rag")

    @patch("app.intent_service.settings")
    def test_support_contact_request_goes_to_support_contacts(self, mock_settings):
        mock_settings.intent_confidence_threshold = 0.65
        decision = resolve_intent_decision(
            IntentResult(intent="support_contact_request", confidence=0.99)
        )
        self.assertEqual(decision, "support_contacts")


class IntentClassifierApiTests(unittest.TestCase):
    def test_normalize_intent_aliases(self):
        self.assertEqual(_normalize_intent("offtopic"), "out_of_scope")
        self.assertEqual(_normalize_intent("it_question"), "it_support_request")
        self.assertEqual(_normalize_intent("support_contacts"), "support_contact_request")
        self.assertEqual(_normalize_intent("support_contact"), "support_contact_request")
        self.assertEqual(_normalize_intent("contact_support"), "support_contact_request")
        self.assertEqual(_normalize_intent("support_ticket"), "it_support_request")

    def test_resolve_reasoning_effort_valid(self):
        self.assertEqual(resolve_reasoning_effort("low"), "low")

    def test_is_no_sampling_model_gpt5(self):
        self.assertTrue(_is_no_sampling_model("gpt-5-nano"))

    def test_is_no_sampling_model_legacy(self):
        self.assertFalse(_is_no_sampling_model("gpt-4o-mini"))

    @patch("app.intent_service.OpenAI")
    def test_gpt5_nano_passes_reasoning_effort(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"intent": "it_support_request", "confidence": 0.9}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        _call_intent_classifier(
            mock_client,
            "gpt-5-nano",
            [{"role": "user", "content": "test"}],
            reasoning_effort="low",
        )

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs.get("reasoning_effort"), "low")

    @patch("app.intent_service.OpenAI")
    def test_gpt4_mini_passes_temperature(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"intent": "greeting", "confidence": 0.8}'
        mock_client.chat.completions.create.return_value = mock_response

        _call_intent_classifier(
            mock_client,
            "gpt-4o-mini",
            [{"role": "user", "content": "привет"}],
            reasoning_effort="low",
        )

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs.get("temperature"), 0)
        self.assertNotIn("reasoning_effort", kwargs)

    @patch("app.intent_service.OpenAI")
    def test_reasoning_effort_retry_without_effort(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"intent": "greeting", "confidence": 0.8}'
        mock_client.chat.completions.create.side_effect = [
            ValueError("unsupported parameter: reasoning_effort"),
            mock_response,
        ]

        raw = _call_intent_classifier(
            mock_client,
            "gpt-5-nano",
            [{"role": "user", "content": "привет"}],
            reasoning_effort="low",
        )
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        second_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        self.assertNotIn("reasoning_effort", second_kwargs)
        self.assertIn("greeting", raw)


class GreetingRoutingIntegrationTests(unittest.TestCase):
    @patch("app.intent_service.OpenAI")
    def test_privet_and_kak_dela_no_rag(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()

        def fake_create(**kwargs):
            user = kwargs["messages"][-1]["content"]
            if user == "привет":
                content = '{"intent": "greeting", "confidence": 0.9}'
            elif user == "как дела":
                content = '{"intent": "greeting", "confidence": 0.0}'
            else:
                content = '{"intent": "it_support_request", "confidence": 0.9}'
            mock_response.choices[0].message.content = content
            return mock_response

        mock_client.chat.completions.create.side_effect = fake_create

        service = IntentService()
        for text in ("привет", "как дела"):
            result = service.classify(text)
            decision = resolve_intent_decision(result)
            self.assertEqual(result.intent, "greeting", msg=text)
            self.assertEqual(decision, "greeting", msg=text)


class SupportContactRoutingIntegrationTests(unittest.TestCase):
    @patch("app.intent_service.OpenAI")
    def test_contact_requests_do_not_route_to_rag(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()

        contact_requests = {
            "как обратиться напрямую",
            "как связаться с поддержкой",
            "дай телефон ИТ",
            "почта поддержки",
            "бот не отправил обращение, куда писать напрямую",
        }

        def fake_create(**kwargs):
            user = kwargs["messages"][-1]["content"]
            if user in contact_requests:
                content = '{"intent": "support_contact_request", "confidence": 0.9}'
            elif user in ("создай заявку", "передай в поддержку", "не работает VPN"):
                content = '{"intent": "it_support_request", "confidence": 0.9}'
            else:
                content = '{"intent": "out_of_scope", "confidence": 0.9}'
            mock_response.choices[0].message.content = content
            return mock_response

        mock_client.chat.completions.create.side_effect = fake_create

        service = IntentService()
        for text in contact_requests:
            result = service.classify(text)
            decision = resolve_intent_decision(result)
            self.assertEqual(result.intent, "support_contact_request", msg=text)
            self.assertEqual(decision, "support_contacts", msg=text)

    @patch("app.intent_service.OpenAI")
    def test_ticket_and_real_it_requests_still_route_to_rag(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()

        def fake_create(**kwargs):
            mock_response.choices[0].message.content = (
                '{"intent": "it_support_request", "confidence": 0.9}'
            )
            return mock_response

        mock_client.chat.completions.create.side_effect = fake_create

        service = IntentService()
        for text in ("создай заявку", "передай в поддержку", "не работает VPN"):
            result = service.classify(text)
            decision = resolve_intent_decision(result)
            self.assertEqual(result.intent, "it_support_request", msg=text)
            self.assertEqual(decision, "rag", msg=text)


class IntentServiceClassifyTests(unittest.TestCase):
    @patch("app.intent_service.OpenAI")
    def test_classify_api_failure_returns_safe_fallback(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("api down")

        service = IntentService()
        result = service.classify("не работает монитор")
        self.assertEqual(result.intent, DEFAULT_INTENT)
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
