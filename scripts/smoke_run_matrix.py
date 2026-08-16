#!/usr/bin/env python3
import json
import sys
import urllib.request
from typing import Callable

API = "http://localhost:8779/message"

CASES = [
    {
        "name": "skill_test_fallback",
        "message": "/run test-fallback hello evo",
        "check": lambda r: "Test skill executed successfully" in r,
    },
    {
        "name": "skill_summarize_url",
        "message": "/run summarize-web-article-key-points https://docs.openclaw.ai",
        "check": lambda r: "Usage:" not in r and ("- " in r or "•" in r),
    },
    {
        "name": "routine_morning_briefing",
        "message": "/run morning briefing",
        "check": lambda r: "☀️ Morning Briefing" in r and "initiated" not in r.lower() and "please wait" not in r.lower(),
    },
]


def call(message: str) -> str:
    body = json.dumps({"message": message, "chat_id": "smoke-prod"}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        data = json.loads(r.read().decode())
    return data.get("reply", "")


def main() -> int:
    failed = []
    for case in CASES:
        print(f"\n=== {case['name']} ===")
        try:
            reply = call(case["message"])
            ok = case["check"](reply)
            print(reply[:2000])
            print(f"RESULT: {'PASS' if ok else 'FAIL'}")
            if not ok:
                failed.append(case["name"])
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(case["name"])
    if failed:
        print("\nFAILED:", ", ".join(failed))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
