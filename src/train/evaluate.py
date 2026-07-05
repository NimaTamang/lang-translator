from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.config import load_config, resolve_path
from src.data.dataset import TranslationDataset
from src.data.zanx_tokenizer import ZanxTokenizer
from src.model.translator import EnToZanxTranslator
from src.train.train import build_dataloader, resolve_encoder_path, run_epoch


def print_samples(
    model: EnToZanxTranslator,
    dataloader: DataLoader,
    en_tokenizer: BertTokenizer,
    zanx_tokenizer: ZanxTokenizer,
    device: torch.device,
    *,
    limit: int = 5,
) -> None:
    model.eval()
    shown = 0
    for batch in dataloader:
        for index in range(batch["en_input_ids"].size(0)):
            if shown >= limit:
                return
            en_ids = batch["en_input_ids"][index].unsqueeze(0).to(device)
            en_mask = batch["en_attention_mask"][index].unsqueeze(0).to(device)
            en_text = en_tokenizer.decode(
                en_ids.squeeze(0).tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_ids = model.translate(
                en_ids,
                en_mask,
                bos_id=zanx_tokenizer.bos_id,
                eos_id=zanx_tokenizer.eos_id,
                max_len=128,
            )
            pred_text = zanx_tokenizer.decode(pred_ids)
            target_text = zanx_tokenizer.decode(batch["zanx_input_ids"][index].tolist())
            print(f"EN:      {en_text}")
            print(f"TARGET:  {target_text}")
            print(f"PRED:    {pred_text}")
            print("---")
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate English to zanX translator.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cpu")
    encoder_path = resolve_encoder_path(config["model"])
    en_tokenizer = BertTokenizer.from_pretrained(
        str(encoder_path),
        local_files_only=encoder_path.exists() and encoder_path.is_dir(),
    )
    zanx_tokenizer = ZanxTokenizer.load(resolve_path("artifacts/tokenizer/zanx"))

    split_path = config["data"]["val_path"] if args.split == "val" else "data/processed/test.jsonl"
    dataloader = build_dataloader(
        resolve_path(split_path),
        en_tokenizer,
        zanx_tokenizer,
        batch_size=config["training"]["batch_size"],
        max_en_len=config["data"]["max_en_len"],
        max_zanx_len=config["data"]["max_zanx_len"],
        shuffle=False,
    )

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    loss = run_epoch(model, dataloader, optimizer, device, train=False)
    print(f"{args.split}_loss={loss:.4f}")
    print_samples(model, dataloader, en_tokenizer, zanx_tokenizer, device, limit=args.samples)


if __name__ == "__main__":
    main()
