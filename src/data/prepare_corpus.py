from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.config import ROOT, resolve_path


def read_pairs(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return pairs
        columns = [part.strip().lower() for part in header.strip().split("\t")]
        try:
            en_idx = columns.index("en")
            zanx_idx = columns.index("zanx")
        except ValueError as exc:
            raise ValueError("TSV header must include 'en' and 'zanx' columns") from exc

        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < max(en_idx, zanx_idx) + 1:
                continue
            en_text = parts[en_idx].strip()
            zanx_text = parts[zanx_idx].strip()
            if en_text and zanx_text:
                pairs.append({"en": en_text, "zanx": zanx_text})
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_pairs(
    pairs: list[dict[str, str]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rows = pairs[:]
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare English-zanX parallel corpus.")
    parser.add_argument(
        "--input",
        default="data/raw/sample_en_zanx_pairs.tsv",
        help="Input TSV with en and zanx columns",
    )
    parser.add_argument("--out", default="data/processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out)
    pairs = read_pairs(input_path)
    if not pairs:
        raise SystemExit(f"No sentence pairs found in {input_path}")

    train_rows, val_rows, test_rows = split_pairs(pairs, seed=args.seed)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(f"Wrote {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test rows to {out_dir}")


if __name__ == "__main__":
    main()
