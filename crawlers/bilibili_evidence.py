"""Bilibili evidence adapters for video URLs/BVIDs and conservative search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
BILIBILI_SEARCH_ALL_API = "https://api.bilibili.com/x/web-interface/search/all/v2"
BVID_PATTERN = re.compile(r"(BV[0-9A-Za-z]{10,})")
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)


@dataclass(slots=True)
class BilibiliVideoEvidence:
    bvid: str
    url: str
    title: str
    author: str
    published_at: str
    text: str
    metrics: dict[str, int]
    raw_json: dict[str, Any]


@dataclass(slots=True)
class BilibiliSearchResult:
    bvid: str
    title: str
    author: str
    play: int
    danmaku: int
    description: str
    raw_json: dict[str, Any]


def extract_bvid(value: str) -> str:
    match = BVID_PATTERN.search(value)
    if not match:
        raise ValueError(f"cannot find BVID in: {value}")
    return match.group(1)


def search_bilibili_videos(
    query: str,
    *,
    limit: int = 10,
    page: int = 1,
    timeout: float = 20.0,
) -> list[BilibiliSearchResult]:
    """Search Bilibili videos using the Agent-Reach-compatible fallback API.

    Agent-Reach currently treats ``/search/all/v2`` as the public fallback when
    ``bili-cli`` is unavailable. The older typed search endpoint can return 412
    in this environment, so this adapter parses the video group from all/v2.
    """
    url = f"{BILIBILI_SEARCH_ALL_API}?{urlencode({'keyword': query, 'page': page})}"
    req = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": "https://search.bilibili.com/",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("code") != 0:
        raise ValueError(f"bilibili search api error: {payload.get('message')}")

    return parse_bilibili_search_results(payload)[: max(0, limit)]


def parse_bilibili_search_results(payload: dict[str, Any]) -> list[BilibiliSearchResult]:
    results: list[BilibiliSearchResult] = []
    for group in (payload.get("data") or {}).get("result") or []:
        if group.get("result_type") != "video":
            continue
        for item in group.get("data") or []:
            bvid = str(item.get("bvid") or "")
            if not bvid:
                continue
            results.append(
                BilibiliSearchResult(
                    bvid=bvid,
                    title=strip_html(str(item.get("title") or "")),
                    author=strip_html(str(item.get("author") or "")),
                    play=_to_int(item.get("play")),
                    danmaku=_to_int(item.get("danmaku")),
                    description=strip_html(str(item.get("description") or "")),
                    raw_json=item,
                )
            )
        break
    return results


def fetch_bilibili_video(value: str, timeout: float = 20.0) -> BilibiliVideoEvidence:
    bvid = extract_bvid(value)
    url = BILIBILI_VIEW_API.format(bvid=bvid)
    req = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("code") != 0:
        raise ValueError(f"bilibili api error for {bvid}: {payload.get('message')}")

    data = payload.get("data") or {}
    owner = data.get("owner") or {}
    stat = data.get("stat") or {}
    metrics = {
        "view": int(stat.get("view") or 0),
        "danmaku": int(stat.get("danmaku") or 0),
        "reply": int(stat.get("reply") or 0),
        "favorite": int(stat.get("favorite") or 0),
        "coin": int(stat.get("coin") or 0),
        "share": int(stat.get("share") or 0),
        "like": int(stat.get("like") or 0),
    }
    title = strip_html(data.get("title") or "")
    description = strip_html(data.get("desc") or "")
    return BilibiliVideoEvidence(
        bvid=bvid,
        url=f"https://www.bilibili.com/video/{bvid}",
        title=title,
        author=str(owner.get("name") or ""),
        published_at=str(data.get("pubdate") or ""),
        text="\n".join(part for part in [title, description] if part),
        metrics=metrics,
        raw_json=data,
    )


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).strip()


def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, str):
        value = value.replace(",", "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
