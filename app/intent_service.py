from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from app.app_logger import logger
from app.config import settings
from app.prompt_loader import load_prompt
from app.text_safety import mask_sensitive_data

VALID_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"},
)
DEFAULT_REASONING_EFFORT = "low"

VALID_INTENTS = frozenset(
    {
        "greeting",
        "out_of_scope",
        "support_contact_request",
        "it_support_request",
    }
)

LEGACY_INTENT_ALIASES = {
    "offtopic": "out_of_scope",
    "it_question": "it_support_request",
    "support_contacts": "support_contact_request",
    "support_contact": "support_contact_request",
    "contact_support": "support_contact_request",
    "support_ticket": "it_support_request",
    "unclear": "it_support_request",
}

DEFAULT_INTENT = "it_support_request"

@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_intent(raw_intent: str) -> str:
    intent = raw_intent.strip().lower()
    intent = LEGACY_INTENT_ALIASES.get(intent, intent)
    if intent in VALID_INTENTS:
        return intent
    return DEFAULT_INTENT


def parse_classifier_response(raw: str) -> IntentResult:
    """Parse model JSON; on failure return safe fallback for RAG."""
    if not raw or not raw.strip():
        logger.warning("Intent classifier returned empty response")
        return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)

    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            logger.warning("Intent classifier returned invalid JSON: %s", raw[:200])
            return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Intent classifier returned invalid JSON: %s", raw[:200])
            return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)

    raw_label = str(payload.get("intent", DEFAULT_INTENT)).strip().lower()
    intent = _normalize_intent(raw_label)
    if raw_label not in VALID_INTENTS and raw_label not in LEGACY_INTENT_ALIASES:
        logger.warning("Intent classifier returned unknown intent: %s", raw_label)

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return IntentResult(intent=intent, confidence=_clamp_confidence(confidence))


def _is_no_sampling_model(model: str) -> bool:
    model_lower = (model or "").lower()
    if "gpt-5" in model_lower:
        return True
    if "reasoning" in model_lower:
        return True
    if model_lower.startswith("o1") or model_lower.startswith("o3"):
        return True
    return False


def resolve_reasoning_effort(value: str | None) -> str:
    raw = (value or DEFAULT_REASONING_EFFORT).strip().lower()
    if raw in VALID_REASONING_EFFORTS:
        return raw
    logger.warning(
        "Invalid INTENT_REASONING_EFFORT=%r, using default %s",
        value,
        DEFAULT_REASONING_EFFORT,
    )
    return DEFAULT_REASONING_EFFORT


def _call_intent_classifier(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
) -> str:
    kwargs: dict = {
        "model": model,
        "messages": messages,
    }
    if not _is_no_sampling_model(model):
        kwargs["temperature"] = 0
    else:
        kwargs["reasoning_effort"] = resolve_reasoning_effort(reasoning_effort)

    include_response_format = True
    include_reasoning_effort = "reasoning_effort" in kwargs

    while True:
        call_kwargs = dict(kwargs)
        if include_response_format:
            call_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**call_kwargs)
            break
        except Exception as exc:
            err = str(exc).lower()
            if include_reasoning_effort and (
                "reasoning_effort" in err
                or ("reasoning" in err and "unsupported" in err)
            ):
                kwargs.pop("reasoning_effort", None)
                include_reasoning_effort = False
                continue
            if include_response_format and ("response_format" in err or "unsupported" in err):
                include_response_format = False
                continue
            raise

    return response.choices[0].message.content or ""


def resolve_intent_decision(result: IntentResult) -> str:
    """Map classification to handler route.

    Only it_support_request (and safe fallbacks) go to RAG. Contact requests
    are answered directly so RAG cannot replace contacts with unrelated context.
    """
    if result.intent == "greeting":
        return "greeting"
    if result.intent == "out_of_scope":
        return "out_of_scope"
    if result.intent == "support_contact_request":
        return "support_contacts"
    return "rag"


class IntentService:
    def __init__(self) -> None:
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.system_prompt = load_prompt(
            settings.intent_prompt_file,
            "",
            "Intent classifier prompt",
        )

    def classify(self, user_text: str) -> IntentResult:
        safe_text = mask_sensitive_data(user_text or "").strip()
        if not safe_text:
            return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": safe_text},
            ]
            raw = _call_intent_classifier(
                self.openai_client,
                settings.intent_model,
                messages,
                reasoning_effort=settings.intent_reasoning_effort,
            )
            return parse_classifier_response(raw)
        except Exception:
            logger.warning("Intent classifier API call failed", exc_info=True)
            return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)
