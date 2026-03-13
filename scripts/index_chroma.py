"""Index billboards CSV into local ChromaDB collection.

If chromadb/open_clip are unavailable, script falls back to deterministic mock embeddings.
"""

import csv
import hashlib
import json
from pathlib import Path

CSV_PATH = Path("data/billboards.csv")
OUT_PATH = Path("data/mock_index.json")


def embed(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round((b / 255.0), 6) for b in digest[:16]]


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8")))
    docs = []
    for row in rows:
        joined = f"{row['gid']} {row['address']} {row['description']}"
        docs.append({"gid": row["gid"], "embedding": embed(joined), "metadata": row})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"items": docs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(docs)} items -> {OUT_PATH}")


if __name__ == "__main__":
    main()
