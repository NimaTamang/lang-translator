from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from src.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download English encoder for offline use.")
    parser.add_argument(
        "--model",
        default="prajjwal1/bert-mini",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--out",
        default="artifacts/models/bert-mini",
        help="Local directory for model files",
    )
    args = parser.parse_args()

    out_dir = resolve_path(args.out)
    snapshot_download(args.model, local_dir=str(out_dir))
    print(f"Downloaded {args.model} to {out_dir}")


if __name__ == "__main__":
    main()
