import re

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3})?[\s\-\(\)]*)?(?:\d[\s\-\(\)]*){9,}")
PASSWORD_RE = re.compile(r"(?i)(пароль|password|pass|pwd)\s*[:=]\s*\S+")
OTP_RE = re.compile(r"(?i)(код|otp|2fa|mfa|одноразовый код)\s*[:=]?\s*\d{4,8}")


def mask_sensitive_data(text: str) -> str:
    if not text:
        return text

    text = PASSWORD_RE.sub(r"\1: [MASKED_SECRET]", text)
    text = OTP_RE.sub(r"\1: [MASKED_CODE]", text)
    text = EMAIL_RE.sub("[MASKED_EMAIL]", text)
    text = PHONE_RE.sub("[MASKED_PHONE]", text)
    return text


def contains_secret_like_data(text: str) -> bool:
    if not text:
        return False

    return bool(PASSWORD_RE.search(text) or OTP_RE.search(text))
