"""
File: train.py

Responsibilities:
- Entry point for training models (mainly DistilBERT) using configuration files.
- Set random seeds and prepare datasets/tokenizer/model.
- Configure Hugging Face Trainer (or custom loop), run training, and save best checkpoints and training logs.

Contributors:
- Anton Nemchinski
- <Name 2>
- <Name 3>

Key functions to implement:
- main() -> None
- setup_training(cfg) -> (model, tokenizer, train_dataset, val_dataset, training_args)
- run_training(model, tokenizer, train_dataset, val_dataset, training_args, cfg) -> None
"""

import argparse
from transformers import TrainingArguments, Trainer
from src.utils import load_config, set_seed, get_device, get_run_id, ensure_dir
from src.models import get_tokenizer, get_distilbert_model
from src.data import load_fake_news_dataset

def main():
    # CLI argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    # Load config and set up
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg)

    ds = load_fake_news_dataset(cfg)
    tokenizer = get_tokenizer(cfg)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg["data"]["max_length"],
        )

    # Tokenize datasets
    ds_tok = ds.map(tokenize_fn, batched=True)
    ds_tok = ds_tok.remove_columns([c for c in ds_tok["train"].column_names if c not in ["input_ids","attention_mask","label"]])
    ds_tok.set_format("torch")

    # Initialize model
    model = get_distilbert_model(cfg).to(device)

    run_id = get_run_id()
    out_dir = f"artifacts/checkpoints/{run_id}"
    ensure_dir(out_dir)

    # ds_tok is a DatasetDict with "train", "validation", "test"
    train_ds = ds_tok["train"]
    eval_ds = ds_tok["validation"]

    if cfg["hardware"]["debug_mode"]:
        # Use a small subset and fewer epochs for quick CPU/GPU tests
        max_n = min(256, len(train_ds))
        train_ds = train_ds.select(range(max_n))
        # You can also override epochs here if you want:
        cfg["train"]["epochs"] = 1

    args_train = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(cfg["train"]["epochs"]),
        per_device_train_batch_size=int(cfg["train"]["batch_train"]),
        per_device_eval_batch_size=int(cfg["train"]["batch_eval"]),
        learning_rate=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=(device.type == "cuda"),
    )


    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(out_dir)

    test_ds = ds_tok["test"]

    test_metrics = trainer.evaluate(eval_dataset=test_ds)  # type: ignore[arg-type]
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    main()
