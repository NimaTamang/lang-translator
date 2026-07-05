from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class ZanxTokenizer:
    """Word-level tokenizer for the zanX target language."""

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2id[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @staticmethod
    def normalize(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token for token in ZanxTokenizer.normalize(text).split(" ") if token]

    def build_vocab(self, sentences: list[str], min_freq: int = 1) -> None:
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(self.tokenize(sentence))

        self.token2id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        next_id = len(SPECIAL_TOKENS)
        for token, freq in sorted(counts.items()):
            if freq >= min_freq and token not in self.token2id:
                self.token2id[token] = next_id
                next_id += 1
        self.id2token = {index: token for token, index in self.token2id.items()}

    def encode(
        self,
        text: str,
        *,
        add_special: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        tokens: list[str] = []
        for token_id in ids:
            if token_id in (self.pad_id, self.bos_id, self.eos_id) and skip_special:
                continue
            token = self.id2token.get(token_id, UNK_TOKEN)
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, directory: str | Path) -> None:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "token2id": self.token2id,
            "special_tokens": SPECIAL_TOKENS,
        }
        with (out_dir / "zanx_vocab.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str | Path) -> ZanxTokenizer:
        path = Path(directory) / "zanx_vocab.json"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        tokenizer = cls()
        tokenizer.token2id = payload["token2id"]
        tokenizer.id2token = {int(index): token for token, index in tokenizer.token2id.items()}
        return tokenizer
