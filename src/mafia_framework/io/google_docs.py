from __future__ import annotations

import re
import urllib.request
from html import unescape
from urllib.parse import urlparse


# Published-doc ids are prefixed with a literal "e/" segment, which looks
# like a path separator -- match that as a unit before falling back to a
# single opaque segment (regular, non-published doc ids).
DOC_ID_RE = re.compile(r"/document/d/(?P<doc_id>e/[^/]+|[^/]+)")


def _published_text_url(doc_url: str) -> str:
    if "output=txt" in doc_url:
        return doc_url

    parsed = urlparse(doc_url)
    if parsed.netloc.endswith("docs.google.com") and "/pub" in parsed.path:
        # Already a published-doc URL (e.g. ".../document/d/e/<id>/pub") --
        # just request the plain-text export directly, without needing to
        # re-derive the doc id (which is error-prone: published ids have a
        # literal "e/" segment that looks like a path separator).
        separator = "&" if parsed.query else "?"
        return f"{doc_url}{separator}output=txt"

    match = DOC_ID_RE.search(doc_url)
    if match:
        doc_id = match.group("doc_id")
        return f"https://docs.google.com/document/d/{doc_id}/pub?output=txt"

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
        # Google now serves published docs wrapped in a large analytics/
        # loader <script> block even when output=txt is requested. Drop
        # script/style blocks entirely (their content isn't document text,
        # unlike ordinary tags which we just unwrap below).
        payload = re.sub(r"(?is)<script.*?</script>", "", payload)
        payload = re.sub(r"(?is)<style.*?</style>", "", payload)
        payload = re.sub(r"(?is)<br\s*/?>", "\n", payload)
        payload = re.sub(r"(?is)</p\s*>", "\n", payload)
        payload = re.sub(r"(?is)<[^>]+>", "", payload)
        payload = unescape(payload)

    return payload.strip()
