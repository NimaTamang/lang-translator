from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.decoder import ZanxDecoder
from src.model.encoder import EnglishEncoder


class EnToZanxTranslator(nn.Module):
    def __init__(
        self,
        encoder_path: str | Path,
        vocab_size: int,
        *,
        freeze_encoder: bool = True,
        decoder_layers: int = 2,
        decoder_dim: int = 256,
        decoder_heads: int = 4,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = EnglishEncoder(encoder_path, freeze=freeze_encoder)
        if self.encoder.hidden_size != decoder_dim:
            self.encoder_proj = nn.Linear(self.encoder.hidden_size, decoder_dim)
        else:
            self.encoder_proj = nn.Identity()
        self.decoder = ZanxDecoder(
            vocab_size,
            d_model=decoder_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            max_len=max_len,
        )
        self.pad_id = 0

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = self.encoder_proj(memory)
        memory_padding = attention_mask == 0
        return memory, memory_padding

    def forward(
        self,
        en_input_ids: torch.Tensor,
        en_attention_mask: torch.Tensor,
        zanx_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_padding = self.encode(en_input_ids, en_attention_mask)
        tgt_padding = zanx_input_ids == self.pad_id
        return self.decoder(
            zanx_input_ids,
            memory,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=memory_padding,
        )

    def compute_loss(
        self,
        en_input_ids: torch.Tensor,
        en_attention_mask: torch.Tensor,
        zanx_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        decoder_input = zanx_input_ids[:, :-1]
        targets = zanx_input_ids[:, 1:]
        logits = self.forward(en_input_ids, en_attention_mask, decoder_input)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_id,
        )

    @torch.no_grad()
    def translate(
        self,
        en_input_ids: torch.Tensor,
        en_attention_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> list[int]:
        self.eval()
        device = en_input_ids.device
        
        # 1. Encode source text once
        memory, memory_padding = self.encode(en_input_ids, en_attention_mask)
        
        # Start sequence with the BOS token
        generated = [bos_id]
        
        # 2. Autoregressive loop
        for _ in range(max_len):
            # Create a 2D tensor with batch dimension [1, current_sequence_length]
            tgt_ids = torch.tensor([generated], device=device, dtype=torch.long)
            
            # Pass to decoder (decoder automatically handles the causal tgt_mask)
            logits = self.decoder(
                tgt_ids, 
                memory, 
                memory_key_padding_mask=memory_padding
            )
            
            # Extract logits for the LAST predicted token position
            next_token_logits = logits[0, -1, :].clone()
            
            # Prevent the model from choosing its own PAD or BOS token during generation
            next_token_logits[self.pad_id] = float('-inf')
            next_token_logits[bos_id] = float('-inf')
            
            # Select the highest probability token ID
            next_id = int(next_token_logits.argmax(dim=-1).item())
            generated.append(next_id)
            
            # Stop if the model predicts the End of Sequence token
            if next_id == eos_id:
                break
                
        # 3. Strip special framing tokens so the tokenizer receives clean text IDs
        clean_generated = [tok for tok in generated if tok not in (bos_id, eos_id, self.pad_id)]
        return clean_generated