"""Text preprocessing utilities for TrustNewsAI."""
import re
from html import unescape

# Regex patterns
WHITESPACE_RE = re.compile(r'\s+')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')


def clean_text(value: str | None) -> str:
    """Clean raw text by removing URLs, emails, HTML entities, and normalizing whitespace."""
    if value is None:
        return ''

    text = unescape(str(value))
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = text.replace('\x00', ' ')
    text = WHITESPACE_RE.sub(' ', text)
    return text.strip()
def build_content(title: str, body: str) -> str:
    """Combine title and body into a single clean content string."""
    return f'{clean_text(title)} {clean_text(body)}'.strip()
