from __future__ import annotations

import re
import urllib.request
from html import unescape
from urllib.parse import urlparse


DOC_ID_RE = re.compile(r"/document/d/(?P<doc_id>[^/]+)")


def _published_text_url(doc_url: str) -> str:
    if "output=txt" in doc_url:
        return doc_url

    match = DOC_ID_RE.search(doc_url)
    if match:
        doc_id = match.group("doc_id")
        return f"https://docs.google.com/document/d/{doc_id}/pub?output=txt"

    parsed = urlparse(doc_url)
    if parsed.netloc.endswith("docs.google.com") and "/pub" in parsed.path:
        separator = "&" if parsed.query else "?"
        return f"{doc_url}{separator}output=txt"

    return doc_url


def fetch_published_google_doc(doc_url: str, *, timeout: int = 20) -> str:
    """Fetch plain text from a published Google Doc URL."""
    text_url = _published_text_url(doc_url)
    request = urllib.request.Request(
        text_url,
        headers={"User-Agent": "mafia-framework/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        charset_match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
        charset = charset_match.group(1) if charset_match else "utf-8"
        payload = response.read().decode(charset, errors="replace")

    if "<html" in payload.lower():
        payload = re.sub(r"(?is)<br\s*/?>", "\n", payload)
        payload = re.sub(r"(?is)</p\s*>", "\n", payload)
        payload = re.sub(r"(?is)<[^>]+>", "", payload)
        payload = unescape(payload)

    return payload.strip()
