from __future__ import annotations

import math

import torch
import torch.nn as nn


class ZanxDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def _positional(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tgt_ids.size(1), device=tgt_ids.device).unsqueeze(0)
        positions = positions.expand(tgt_ids.size(0), -1)
        return self.embedding(tgt_ids) * math.sqrt(self.d_model) + self.pos_embedding(positions)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt_ids.size(1), device=tgt_ids.device)
        hidden = self._positional(tgt_ids)
        decoded = self.transformer_decoder(
            hidden,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output(decoded)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)
