from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel


class EnglishEncoder(nn.Module):
    def __init__(self, model_path: str | Path, *, freeze: bool = True) -> None:
        super().__init__()
        path = Path(model_path)
        local_only = path.exists()
        self.bert = BertModel.from_pretrained(str(path), local_files_only=local_only)
        self.hidden_size = self.bert.config.hidden_size
        if freeze:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
