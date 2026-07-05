from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn

from src.config import load_config, resolve_path
from src.data.zanx_tokenizer import ZanxTokenizer
from src.model.translator import EnToZanxTranslator
from src.train.train import resolve_encoder_path


class EncoderOnnxWrapper(nn.Module):
    def __init__(self, model: EnToZanxTranslator) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        memory, _ = self.model.encode(input_ids, attention_mask)
        return memory


class DecoderStepOnnxWrapper(nn.Module):
    def __init__(self, model: EnToZanxTranslator) -> None:
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            tgt_ids,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


def export_module(
    module: nn.Module,
    out_path: Path,
    sample_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    *,
    opset: int,
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
        "do_constant_folding": True,
    }
    try:
        # Use legacy exporter to keep dynamic_axes behavior stable across PyTorch releases.
        torch.onnx.export(module, sample_inputs, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older versions may not expose the dynamo flag.
        torch.onnx.export(module, sample_inputs, str(out_path), **export_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export translator to ONNX v2 bundle.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--out", default="artifacts/releases/v2")
    parser.add_argument("--version", default="v2")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cpu")
    encoder_path = resolve_encoder_path(config["model"])
    zanx_tokenizer = ZanxTokenizer.load(resolve_path("artifacts/tokenizer/zanx"))

    model = EnToZanxTranslator(
        encoder_path,
        zanx_tokenizer.vocab_size,
        freeze_encoder=config["model"]["freeze_encoder"],
        decoder_layers=config["model"]["decoder_layers"],
        decoder_dim=config["model"]["decoder_dim"],
        decoder_heads=config["model"]["decoder_heads"],
        max_len=config["data"]["max_zanx_len"],
    ).to(device)
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_dir = resolve_path(args.out)
    encoder_onnx = out_dir / f"encoder-{args.version}.onnx"
    decoder_onnx = out_dir / f"decoder-step-{args.version}.onnx"

    batch, seq = 1, 8
    tgt_len = 4
    input_ids = torch.ones(batch, seq, dtype=torch.long)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    tgt_ids = torch.ones(batch, tgt_len, dtype=torch.long)
    memory = torch.randn(batch, seq, config["model"]["decoder_dim"])
    memory_mask = torch.zeros(batch, seq, dtype=torch.bool)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float("-inf"), diagonal=1)

    export_module(
        EncoderOnnxWrapper(model),
        encoder_onnx,
        (input_ids, attention_mask),
        ["input_ids", "attention_mask"],
        ["memory"],
        opset=args.opset,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "memory": {0: "batch", 1: "sequence"},
        },
    )
    export_module(
        DecoderStepOnnxWrapper(model),
        decoder_onnx,
        (tgt_ids, memory, memory_mask, tgt_mask),
        ["tgt_ids", "memory", "memory_key_padding_mask", "tgt_mask"],
        ["logits"],
        opset=args.opset,
        dynamic_axes={
            "tgt_ids": {0: "batch", 1: "tgt_sequence"},
            "memory": {0: "batch", 1: "sequence"},
            "memory_key_padding_mask": {0: "batch", 1: "sequence"},
            "tgt_mask": {0: "tgt_sequence", 1: "tgt_sequence"},
            "logits": {0: "batch", 1: "tgt_sequence"},
        },
    )

    shutil.copy2(
        resolve_path("artifacts/tokenizer/zanx/zanx_vocab.json"),
        out_dir / "zanx_vocab.json",
    )
    en_tokenizer_dir = out_dir / "en_tokenizer"
    if encoder_path.exists():
        if en_tokenizer_dir.exists():
            shutil.rmtree(en_tokenizer_dir)
        shutil.copytree(encoder_path, en_tokenizer_dir)

    manifest = {
        "model_version": args.version,
        "format": "onnx",
        "opset": args.opset,
        "encoder": config["model"]["encoder_name"],
        "encoder_onnx": encoder_onnx.name,
        "decoder_onnx": decoder_onnx.name,
        "decoder_layers": config["model"]["decoder_layers"],
        "max_seq_len": config["data"]["max_en_len"],
        "created": str(date.today()),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Exported ONNX bundle to {out_dir}")


if __name__ == "__main__":
    main()
