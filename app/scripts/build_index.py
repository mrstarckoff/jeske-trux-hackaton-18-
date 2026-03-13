from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


def pseudo_embed(path: str, dim: int = 128) -> list[float]:
    seed = hashlib.sha256(path.encode('utf-8')).digest()
    vec = [seed[i % len(seed)] / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def main() -> None:
    parser = argparse.ArgumentParser(description='Build CLIP-like embeddings for CSV records')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    with inp.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gid', 'address', 'latitude', 'longitude', 'embedding'])
        writer.writeheader()
        for row in rows:
            emb = pseudo_embed(row.get('image_path', row.get('gid', '')))
            writer.writerow(
                {
                    'gid': row.get('gid', ''),
                    'address': row.get('address', ''),
                    'latitude': row.get('latitude', ''),
                    'longitude': row.get('longitude', ''),
                    'embedding': ' '.join(f'{v:.8f}' for v in emb),
                }
            )


if __name__ == '__main__':
    main()
