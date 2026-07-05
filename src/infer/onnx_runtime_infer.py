from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer

from src.config import resolve_path
from src.data.zanx_tokenizer import ZanxTokenizer


def causal_mask(size: int) -> np.ndarray:
    mask = np.triu(np.ones((size, size), dtype=np.float32), k=1)
    mask[mask == 1.0] = -np.inf
    return mask


class OnnxTranslator:
    def __init__(self, bundle_dir: str | Path) -> None:
        bundle = Path(bundle_dir)
        with (bundle / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.zanx_tokenizer = ZanxTokenizer.load(bundle)
        self.en_tokenizer = BertTokenizer.from_pretrained(bundle / "en_tokenizer")
        self.bos_id = self.zanx_tokenizer.bos_id
        self.eos_id = self.zanx_tokenizer.eos_id

        encoder_path = bundle / manifest["encoder_onnx"]
        decoder_path = bundle / manifest["decoder_onnx"]
        self.encoder = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

    def translate(self, text: str, *, max_len: int = 128) -> str:
        encoding = self.en_tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        memory = self.encoder.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        memory_mask = attention_mask == 0

        generated = [self.bos_id]
        for _ in range(max_len):
            tgt_len = len(generated)
            tgt_ids = np.array([generated], dtype=np.int64)
            tgt_mask = causal_mask(tgt_len)
            logits = self.decoder.run(
                None,
                {
                    "tgt_ids": tgt_ids,
                    "memory": memory,
                    "memory_key_padding_mask": memory_mask,
                    "tgt_mask": tgt_mask,
                },
            )[0]
            next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            generated.append(next_id)
            if next_id == self.eos_id:
                break
        return self.zanx_tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX English to zanX inference.")
    parser.add_argument("--bundle", default="artifacts/releases/v2")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    translator = OnnxTranslator(resolve_path(args.bundle))
    print(translator.translate(args.text))


if __name__ == "__main__":
    main()
