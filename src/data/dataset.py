from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.data.zanx_tokenizer import ZanxTokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        en_tokenizer: BertTokenizer,
        zanx_tokenizer: ZanxTokenizer,
        *,
        max_en_len: int = 128,
        max_zanx_len: int = 128,
    ) -> None:
        self.rows = self._load_rows(jsonl_path)
        self.en_tokenizer = en_tokenizer
        self.zanx_tokenizer = zanx_tokenizer
        self.max_en_len = max_en_len
        self.max_zanx_len = max_zanx_len

    @staticmethod
    def _load_rows(path: str | Path) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        en_encoding = self.en_tokenizer(
            row["en"],
            max_length=self.max_en_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        zanx_ids = self.zanx_tokenizer.encode(row["zanx"], add_special=True, max_len=self.max_zanx_len)
        return {
            "en_input_ids": en_encoding["input_ids"].squeeze(0),
            "en_attention_mask": en_encoding["attention_mask"].squeeze(0),
            "zanx_input_ids": torch.tensor(zanx_ids, dtype=torch.long),
        }
