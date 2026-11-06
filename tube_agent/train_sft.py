"""Small text-state LoRA SFT; loss is applied only to the next assistant action."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path

from .model_utils import DEFAULT_MODEL, resolve_model


def make_examples(records, tokenizer, max_length):
    examples, skipped = [], 0
    for record in records:
        if record["source"]["split"] != "train" or record["source"]["policy"] != "model":
            raise ValueError("Training requires self-generated TRAIN episodes")
        messages = record["messages"]
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(messages[:i], tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=False)
            if isinstance(prompt, Mapping):
                prompt = prompt["input_ids"]
            answer = tokenizer.encode(message["content"] + tokenizer.eos_token, add_special_tokens=False)
            if len(prompt) + len(answer) > max_length:
                skipped += 1
                continue
            examples.append({"input_ids": prompt + answer,
                             "attention_mask": [1]*(len(prompt)+len(answer)),
                             "labels": [-100]*len(prompt)+answer})
    if not examples:
        raise ValueError("No trainable actions after length filtering")
    return examples, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("checkpoints/tube-lora"))
    p.add_argument("--epochs", type=float, default=1.)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output directory must be empty to preserve prior experiments")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError("Run training on the GPU instance")
    torch.manual_seed(42)
    model_path = str(resolve_model(args.model))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples, skipped = make_examples(records, tokenizer, args.max_length)
    print(f"Training actions={len(examples)}; skipped overlong actions={skipped}", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and ".language_model." in name]
    if not targets:
        raise RuntimeError("Could not identify language-model LoRA targets")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2*args.rank,
                                            target_modules=targets, lora_dropout=.05, task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.print_trainable_parameters()

    def collate(batch):
        width = max(len(x["input_ids"]) for x in batch)
        padding = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "labels": -100}
        return {key: torch.tensor([x[key]+[pad]*(width-len(x[key])) for x in batch], dtype=torch.long)
                for key, pad in padding.items()}

    training = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=args.learning_rate, bf16=True, logging_steps=5,
        save_strategy="epoch", save_total_limit=2, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, remove_unused_columns=False, seed=42)
    trainer = Trainer(model=model, args=training, train_dataset=examples, data_collator=collate)
    trainer.train()
    trainer.save_model(str(args.output))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output))
        metadata = {"stage": "self_success_sft", "episodes": len(records), "actions": len(examples),
                    "overlong_actions_skipped": skipped, "base_model": model_path,
                    "vision_encoder_trained": False, "grpo": False}
        (args.output/"experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
