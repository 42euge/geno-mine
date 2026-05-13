"""Privacy filters — scrub sensitive data from training examples.

1. Path scrubbing: absolute paths → relative
2. Secret detection: API keys, tokens, bearer tokens
3. PII removal: emails, phone numbers
4. Content filtering: skip segments touching sensitive files
"""

from __future__ import annotations

import re
from pathlib import Path

_HOME = str(Path.home())

SECRET_PATTERNS = [
    re.compile(r"(sk-|ghp_|ghs_|AKIA|xox[bopsa]-)[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer [A-Za-z0-9._\-]{20,}"),
    re.compile(r"token[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._\-]{20,}[\"']", re.IGNORECASE),
    re.compile(r"password[\"']?\s*[:=]\s*[\"'][^\"']{4,}[\"']", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
]

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

SENSITIVE_PATHS = [".env", "credentials", ".ssh/", ".aws/", "secrets"]


def scrub_paths(text: str) -> str:
    """Replace absolute home-directory paths with relative ones."""
    return text.replace(_HOME, "~")


def scrub_secrets(text: str) -> str:
    """Redact API keys, tokens, and passwords."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("<REDACTED>", text)
    return text


def scrub_pii(text: str) -> str:
    """Replace email addresses and phone numbers with placeholders."""
    text = EMAIL_PATTERN.sub("<EMAIL>", text)
    text = PHONE_PATTERN.sub("<PHONE>", text)
    return text


def has_sensitive_content(text: str) -> bool:
    """Check if text references sensitive files."""
    lower = text.lower()
    return any(s in lower for s in SENSITIVE_PATHS)


def scrub(text: str, *, paths: bool = True, secrets: bool = True, pii: bool = True) -> str:
    """Apply all scrubbing filters."""
    if paths:
        text = scrub_paths(text)
    if secrets:
        text = scrub_secrets(text)
    if pii:
        text = scrub_pii(text)
    return text
