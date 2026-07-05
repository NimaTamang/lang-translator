import torch
from transformers import BertTokenizer
from src.config import load_config, resolve_path
from src.data.zanx_tokenizer import ZanxTokenizer
from src.model.translator import EnToZanxTranslator
from src.train.train import resolve_encoder_path

def main():
    # 1. Setup configurations and device
    config = load_config("configs/dev.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load tokenizers
    encoder_path = resolve_encoder_path(config["model"])
    en_tokenizer = BertTokenizer.from_pretrained(str(encoder_path))
    zanx_tokenizer = ZanxTokenizer.load(resolve_path("artifacts/tokenizer/zanx"))
    
    # 3. Initialize model and load trained weights
    # FIX: Overriding config with 64 to precisely match your 'best.pt' checkpoint size
    model_max_len = 64 
    
    model = EnToZanxTranslator(
        encoder_path,
        zanx_tokenizer.vocab_size,
        freeze_encoder=config["model"]["freeze_encoder"],
        decoder_layers=config["model"]["decoder_layers"],
        decoder_dim=config["model"]["decoder_dim"],
        decoder_heads=config["model"]["decoder_heads"],
        max_len=model_max_len,
    ).to(device)
    
    checkpoint = torch.load(resolve_path("artifacts/checkpoints/best.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode
    
    print("\n🤖 Translator Ready! Type 'quit' or 'exit' to stop.\n")
    
    # 4. Interactive loop
    while True:
        try:
            user_input = input("Enter English text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                break
                
            # FIX: Added required padding, truncation, and max_length limits 
            # to prevent model attention vectors from collapsing into single words.
            encoded = en_tokenizer(
                user_input, 
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=model_max_len
            )
            en_ids = encoded["input_ids"].to(device)
            en_mask = encoded["attention_mask"].to(device)
            
            # Generate translation
            with torch.no_grad(): # Disable gradient calculation for speed/memory
                # FIX: Set max_len to 64 to prevent out-of-bounds generation errors
                pred_ids = model.translate(
                    en_ids,
                    en_mask,
                    bos_id=zanx_tokenizer.bos_id,
                    eos_id=zanx_tokenizer.eos_id,
                    max_len=model_max_len,
                )
            
            # Decode and print output
            pred_text = zanx_tokenizer.decode(pred_ids)
            print(f"👉 zanX Output: {pred_text}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
