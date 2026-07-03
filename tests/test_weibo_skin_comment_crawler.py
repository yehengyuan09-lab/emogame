import unittest

import httpx

from crawlers.weibo_skin_comment_crawler import (
    CrawlEntry,
    LONG_TEXT_URL,
    WeiboPost,
    WeiboAccessBlocked,
    WeiboClient,
    fetch_post_text,
    format_output,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def _request(self, url, params=None):
        self.calls.append((url, params))
        return self.response


class FetchPostTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_expands_long_post_and_cleans_html(self):
        client = FakeClient({
            "ok": 1,
            "data": {"longTextContent": "<b>full</b> post &amp; details"},
        })

        text = await fetch_post_text(client, {
            "id": "123",
            "isLongText": True,
            "text": "preview…",
        })

        self.assertEqual(text, "full post & details")
        self.assertEqual(client.calls, [(LONG_TEXT_URL, {"id": "123"})])

    async def test_short_post_does_not_request_extension(self):
        client = FakeClient({})

        text = await fetch_post_text(client, {
            "id": "123",
            "isLongText": False,
            "text": "<b>complete</b>",
        })

        self.assertEqual(text, "complete")
        self.assertEqual(client.calls, [])

    async def test_empty_extension_falls_back_to_preview(self):
        client = FakeClient({"ok": 1, "data": {}})

        text = await fetch_post_text(client, {
            "id": "123",
            "isLongText": True,
            "text": "preview…",
        })

        self.assertEqual(text, "preview…")


class WeiboClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_432_is_not_retried(self):
        request_count = 0

        def handler(request):
            nonlocal request_count
            request_count += 1
            return httpx.Response(432, request=request)

        client = WeiboClient("cookie", max_retries=4)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def no_wait():
            return None

        client.rate_limiter.wait = no_wait
        try:
            with self.assertRaisesRegex(WeiboAccessBlocked, "HTTP 432"):
                await client._request("https://m.weibo.cn/api/container/getIndex")
        finally:
            await client._client.aclose()

        self.assertEqual(request_count, 1)


class OutputTests(unittest.TestCase):
    def test_format_output_keeps_full_title(self):
        full_text = "x" * 100
        post = WeiboPost("123", 1, "user", full_text, "", 0, 0, 0)
        entry = CrawlEntry("123", full_text, 0, 0, 0, post, [])

        output = format_output([entry])

        self.assertEqual(output[0]["title"], full_text)


if __name__ == "__main__":
    unittest.main()
