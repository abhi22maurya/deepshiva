#!/usr/bin/env python3
"""
DeepShiva Training Script

This script handles the training of the DeepShiva MoE model with support for:
- Mixture of Experts (MoE) architecture
- DeepSpeed ZeRO-3 optimization
- LoRA fine-tuning
- Mixed precision training
- Checkpointing and resuming
"""

import os
import yaml
import torch
import wandb
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    get_scheduler,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import get_last_checkpoint

# Custom MoE Model Implementation
from models.moe_model import DeepShivaMoE
from data.data_processor import load_and_process_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepShiva MoE model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/moe_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load and validate the configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict):
    """Initialize Weights & Biases for experiment tracking."""
    if config["logging"]["report_to"] == "wandb":
        wandb.init(
            project="deepshiva",
            name=config["logging"].get("run_name", "deepshiva-moe"),
            config=config,
        )


def get_model(config: Dict) -> torch.nn.Module:
    """Initialize or load the MoE model."""
    model_config = config["model"]
    
    # Initialize from scratch
    model = DeepShivaMoE(
        num_layers=model_config["num_layers"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config.get("num_key_value_heads", model_config["num_attention_heads"]),
        num_experts=model_config["num_experts"],
        num_selected_experts=model_config["num_selected_experts"],
        vocab_size=model_config["vocab_size"],
        max_position_embeddings=model_config["max_position_embeddings"],
        rms_norm_eps=model_config["rms_norm_eps"],
        rope_theta=model_config.get("rope_theta", 10000.0),
        use_parallel_residual=model_config.get("use_parallel_residual", True),
        moe_layer_frequency=model_config.get("moe_layer_frequency", 1),
    )
    
    return model


def setup_lora(model: torch.nn.Module, config: Dict) -> torch.nn.Module:
    """Apply LoRA to the model if specified in config."""
    if "lora" in config:
        lora_config = config["lora"]
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model


def setup_training_args(config: Dict, output_dir: str) -> TrainingArguments:
    """Set up training arguments from config."""
    train_config = config["training"]
    logging_config = config["logging"]
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_config["micro_batch_size"],
        per_device_eval_batch_size=train_config["micro_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_steps"],
        max_steps=train_config["max_steps"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        logging_steps=logging_config.get("logging_steps", 10),
        save_steps=logging_config["save_steps"],
        eval_steps=logging_config["eval_steps"],
        save_total_limit=logging_config["save_total_limit"],
        report_to=logging_config["report_to"],
        run_name=logging_config["run_name"],
        ddp_find_unused_parameters=False,
    )


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Set up distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize wandb if enabled
    if config["logging"]["report_to"] == "wandb" and args.local_rank <= 0:
        setup_wandb(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = get_model(config)
    
    # Apply LoRA if specified
    model = setup_lora(model, config)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_process_dataset(config, tokenizer)
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Set up training arguments
    training_args = setup_training_args(
        config,
        output_dir=config["logging"]["checkpoint_dir"],
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    if args.local_rank <= 0:
        trainer.save_model(os.path.join(training_args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "final"))


if __name__ == "__main__":
    main()
