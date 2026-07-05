from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import resolve_path
from src.data.zanx_tokenizer import ZanxTokenizer


def load_zanx_sentences(*paths: Path) -> list[str]:
    sentences: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    row = json.loads(line)
                    sentences.append(row["zanx"])
                else:
                    sentences.append(line)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zanX word-level tokenizer vocabulary.")
    parser.add_argument(
        "--input",
        default="data/processed/train.jsonl",
        help="Monolingual zanX text file or JSONL with zanx field",
    )
    parser.add_argument("--out", default="artifacts/tokenizer/zanx", help="Output directory")
    parser.add_argument("--min-freq", type=int, default=1)
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out)
    sentences = load_zanx_sentences(input_path)
    if not sentences:
        raise SystemExit(f"No zanX sentences found in {input_path}")

    tokenizer = ZanxTokenizer()
    tokenizer.build_vocab(sentences, min_freq=args.min_freq)
    tokenizer.save(out_dir)
    print(f"Saved zanX tokenizer with {tokenizer.vocab_size} tokens to {out_dir}")


if __name__ == "__main__":
    main()
