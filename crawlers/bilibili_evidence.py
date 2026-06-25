"""Bilibili evidence adapter for known video URLs/BVIDs.

This module intentionally does not implement broad search crawling. The first
stage research workflow is: manually identify relevant videos, then fetch their
stable metadata and engagement metrics by BVID.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
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


def extract_bvid(value: str) -> str:
    match = BVID_PATTERN.search(value)
    if not match:
        raise ValueError(f"cannot find BVID in: {value}")
    return match.group(1)


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
