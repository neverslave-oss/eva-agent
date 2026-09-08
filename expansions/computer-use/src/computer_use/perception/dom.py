from __future__ import annotations

import re


def snapshot_dom(html: str = "", url: str = "") -> dict:
    """Create a compact DOM snapshot summary from html/url inputs."""
    title = ""
    if html:
        m = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()

    refs = re.findall(r"\b(id|name)=['\"]([^'\"]+)['\"]", html, flags=re.IGNORECASE) if html else []
    return {
        "kind": "dom",
        "url": url or None,
        "title": title or None,
        "ref_count": len(refs),
    }
