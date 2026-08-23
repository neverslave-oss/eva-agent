#!/usr/bin/env python3
"""
embed.py
========
Embed the unembedded ActivityEvent rows in `raw_seed.jsonl` using the local
embeddings server (`:8770`, embeddinggemma-300m → 768-d) and write an
embedded copy to `embedded_seed.jsonl`.

This is the step that turns the seed text into the numeric feature vectors
that the clustering layer (cluster_v0.py) consumes.

Writes each row with intent_embedding (768 floats) + embedding_model/dim set.
Skips rows already embedded (idempotent).
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

EMBED_URL = "http://localhost:8770/embeddings"
EMBED_MODEL = "embeddinggemma-300m"
EMBED_DIM = 768


def embed_texts(texts: list[str], batch: int = 8) -> list[list[float]]:
    """Embed a list of strings via the local server, batching.""" 
    if requests is None:
        raise SystemExit("requires `requests` — pip install requests")
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        # Empty strings break the model; substitute a neutral token.
        chunk = [t if t.strip() else "no activity recorded" for t in chunk]
        r = requests.post(EMBED_URL, json={"input": chunk}, timeout=120)
        r.raise_for_status()
        data = r.json().get("data", [])
        out.extend(data)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(os.path.dirname(__file__), "datasets", "raw_seed.jsonl"))
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(__file__), "datasets", "embedded_seed.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="max rows to embed (dev)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8")]
    to_embed = [r for r in rows if not r.get("intent_embedding")]
    if args.limit:
        to_embed = to_embed[: args.limit]
        rows = rows[: args.limit]  # keep parallel

    if not to_embed:
        print("No rows need embedding.")
        return

    # Build the text we embed: intent + intent_text (concise vector of meaning).
    texts = [f"{r['intent']} {r['intent_text']}".strip() for r in to_embed]

    print(f"Embedding {len(texts)} rows via {EMBED_URL} ...")
    vectors = embed_texts(texts)
    assert len(vectors) == len(to_embed), "embed count mismatch"

    for row, vec in zip(to_embed, vectors):
        row["intent_embedding"] = vec
        row["embedding_dim"] = len(vec)
        row["embedding_model"] = EMBED_MODEL

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(vectors)} embedded rows -> {args.out}")


if __name__ == "__main__":
    main()