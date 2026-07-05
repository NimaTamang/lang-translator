from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.config import load_config, resolve_path
from src.data.dataset import TranslationDataset
from src.data.zanx_tokenizer import ZanxTokenizer
from src.model.translator import EnToZanxTranslator


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_encoder_path(model_cfg: dict) -> Path:
    local_path = resolve_path(model_cfg["encoder_local_path"])
    if local_path.exists():
        return local_path
    return Path(model_cfg["encoder_name"])


def build_dataloader(
    path: Path,
    en_tokenizer: BertTokenizer,
    zanx_tokenizer: ZanxTokenizer,
    *,
    batch_size: int,
    max_en_len: int,
    max_zanx_len: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TranslationDataset(
        path,
        en_tokenizer,
        zanx_tokenizer,
        max_en_len=max_en_len,
        max_zanx_len=max_zanx_len,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: EnToZanxTranslator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train: bool,
    grad_accum: int = 1,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader, start=1):
        en_input_ids = batch["en_input_ids"].to(device)
        en_attention_mask = batch["en_attention_mask"].to(device)
        zanx_input_ids = batch["zanx_input_ids"].to(device)

        if train:
            loss = model.compute_loss(en_input_ids, en_attention_mask, zanx_input_ids)
            loss = loss / grad_accum
            loss.backward()
            if step % grad_accum == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                loss = model.compute_loss(en_input_ids, en_attention_mask, zanx_input_ids)

        total_loss += float(loss.item()) * (grad_accum if train else 1)
        steps += 1

    return total_loss / max(steps, 1)


def save_checkpoint(
    path: Path,
    model: EnToZanxTranslator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train English to zanX translator.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="artifacts/checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cpu")

    encoder_path = resolve_encoder_path(config["model"])
    en_tokenizer = BertTokenizer.from_pretrained(
        str(encoder_path),
        local_files_only=encoder_path.exists() and encoder_path.is_dir(),
    )
    zanx_tokenizer = ZanxTokenizer.load(resolve_path("artifacts/tokenizer/zanx"))

    train_loader = build_dataloader(
        resolve_path(config["data"]["train_path"]),
        en_tokenizer,
        zanx_tokenizer,
        batch_size=config["training"]["batch_size"],
        max_en_len=config["data"]["max_en_len"],
        max_zanx_len=config["data"]["max_zanx_len"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        resolve_path(config["data"]["val_path"]),
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

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["training"]["learning_rate"])

    output_dir = resolve_path(args.output)
    best_path = output_dir / "best.pt"
    best_val = float("inf")
    patience = config["training"]["early_stopping_patience"]
    stale_epochs = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            grad_accum=config["training"]["gradient_accumulation"],
        )
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_loss, config)
            print(f"saved checkpoint to {best_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("early stopping")
                break

    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"best_val_loss": best_val}, handle, indent=2)


if __name__ == "__main__":
    main()
