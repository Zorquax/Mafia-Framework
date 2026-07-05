import unittest
from unittest.mock import Mock, patch

from mafia_framework.io.google_docs import _published_text_url, fetch_published_google_doc

PUBLISHED_URL = (
    "https://docs.google.com/document/d/e/2PACX-1vSPQZvpSczQsv5_PndDHt14nUv3atE8cZ4J0tUlKK4"
    "dcrxow9AKIS74AJy1Bg047pBEVfWN6UWzdz3g/pub"
)


class TestPublishedTextUrl(unittest.TestCase):

    def test_published_doc_url_keeps_full_id_with_e_prefix(self):
        # Published-doc ids look like "e/<long-id>", where the "e/" is part
        # of the id itself, not a path separator. Naively grabbing everything
        # up to the next "/" truncates the id down to just "e" and produces
        # a URL that 404s.
        result = _published_text_url(PUBLISHED_URL)
        self.assertEqual(result, PUBLISHED_URL + "?output=txt")

    def test_published_doc_url_with_existing_query_string(self):
        result = _published_text_url(PUBLISHED_URL + "?foo=bar")
        self.assertEqual(result, PUBLISHED_URL + "?foo=bar&output=txt")

    def test_editor_url_is_converted_to_published_text_url(self):
        result = _published_text_url("https://docs.google.com/document/d/1AbCdEfGhIjKlmnop/edit")
        self.assertEqual(result, "https://docs.google.com/document/d/1AbCdEfGhIjKlmnop/pub?output=txt")

    def test_url_already_requesting_output_txt_is_untouched(self):
        url = PUBLISHED_URL + "?output=txt"
        self.assertEqual(_published_text_url(url), url)


class TestFetchPublishedGoogleDoc(unittest.TestCase):

    def test_strips_script_and_style_blocks_before_extracting_text(self):
        # Google wraps published docs in a large analytics/loader <script>
        # block even when output=txt is requested. If script contents aren't
        # dropped before generic tag-stripping, the JS code itself leaks
        # into the "plain text" output ahead of the real document content.
        html = (
            "<!DOCTYPE html><html><head>"
            "<script>var x = 1; if (x < 2) { doStuff(); }</script>"
            "<style>.c0 { color: red; }</style>"
            "</head><body>"
            "<p><span>Alice: I think Bob is scum</span></p>"
            "<p><span>Bob was eliminated!</span></p>"
            "</body></html>"
        )
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.read.return_value = html.encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            text = fetch_published_google_doc(PUBLISHED_URL)

        self.assertNotIn("doStuff", text)
        self.assertNotIn("color: red", text)
        self.assertIn("Alice: I think Bob is scum", text)
        self.assertIn("Bob was eliminated!", text)
