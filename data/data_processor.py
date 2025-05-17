"""
Data Processing Pipeline for DeepShiva

This module handles data loading, preprocessing, and tokenization for training
the DeepShiva MoE model. It supports:
- Loading and processing code datasets (HumanEval, MBPP, etc.)
- Loading and processing multilingual datasets
- Efficient tokenization and batching
- Data augmentation techniques
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor for DeepShiva model training and evaluation.
    
    Handles loading, preprocessing, and tokenizing datasets for training
    and evaluation of the DeepShiva MoE model.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        data_dir: str = "data",
        cache_dir: Optional[str] = None,
        preprocessing_num_workers: int = 4,
    ):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: Tokenizer to use for processing text
            max_seq_length: Maximum sequence length for model inputs
            data_dir: Directory containing data files
            cache_dir: Directory to cache processed datasets
            preprocessing_num_workers: Number of workers for preprocessing
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.preprocessing_num_workers = preprocessing_num_workers
    
    def load_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs,
    ) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Optional dataset split to load
            streaming: Whether to stream the dataset
            **kwargs: Additional arguments to pass to load_dataset
            
        Returns:
            Loaded dataset
        """
        # Check if it's a local dataset
        local_path = os.path.join(self.data_dir, "raw", f"{dataset_name}.json")
        if os.path.exists(local_path):
            logger.info(f"Loading local dataset from {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert to Dataset format
            dataset = Dataset.from_dict({
                key: [item[key] for item in data] 
                for key in data[0].keys()
            })
            
            # Split if needed
            if split is not None:
                dataset = self._split_dataset(dataset, split)
            
            return dataset
        
        # Otherwise load from Hugging Face
        logger.info(f"Loading dataset {dataset_name} from Hugging Face")
        return load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=self.cache_dir,
            **kwargs,
        )
    
    def _split_dataset(
        self,
        dataset: Dataset,
        split: str,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> Dataset:
        """Split a dataset into train/validation/test."""
        if split == "train_test":
            return dataset.train_test_split(test_size=1-train_ratio, seed=seed)
        elif split == "train":
            return dataset.train_test_split(test_size=1-train_ratio, seed=seed)["train"]
        elif split == "test":
            return dataset.train_test_split(test_size=1-train_ratio, seed=seed)["test"]
        else:
            raise ValueError(f"Unsupported split: {split}")
    
    def preprocess_code_dataset(
        self,
        dataset: Dataset,
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
        prompt_template: Optional[str] = None,
    ) -> Dataset:
        """
        Preprocess a code dataset (like HumanEval or MBPP).
        
        Args:
            dataset: The dataset to preprocess
            instruction_column: Column containing the instruction
            input_column: Column containing the input code
            output_column: Column containing the expected output
            prompt_template: Optional template for formatting prompts
            
        Returns:
            Preprocessed dataset
        """
        def format_prompt(example):
            instruction = example.get(instruction_column, "")
            code_input = example.get(input_column, "")
            
            if prompt_template:
                if code_input:
                    prompt = prompt_template.format(
                        instruction=instruction,
                        input=code_input,
                    )
                else:
                    prompt = prompt_template.format(
                        instruction=instruction,
                        input="",
                    )
            else:
                # Default template
                if code_input:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{code_input}\n\n### Response:\n"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            return {
                "prompt": prompt,
                "completion": example[output_column],
            }
        
        return dataset.map(
            format_prompt,
            num_proc=self.preprocessing_num_workers,
            remove_columns=dataset.column_names,
        )
    
    def preprocess_multilingual_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        language_column: Optional[str] = "language",
    ) -> Dataset:
        """
        Preprocess a multilingual dataset.
        
        Args:
            dataset: The dataset to preprocess
            text_column: Column containing the text
            language_column: Optional column containing the language code
            
        Returns:
            Preprocessed dataset
        """
        def format_multilingual(example):
            text = example[text_column]
            language = example.get(language_column, None)
            
            # Add language tag if available
            if language:
                formatted_text = f"<{language}>\n{text}"
            else:
                formatted_text = text
                
            return {"text": formatted_text}
        
        return dataset.map(
            format_multilingual,
            num_proc=self.preprocessing_num_workers,
            remove_columns=[col for col in dataset.column_names if col != text_column and col != language_column],
        )
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        prompt_column: Optional[str] = None,
        completion_column: Optional[str] = None,
        add_eos_token: bool = True,
    ) -> Dataset:
        """
        Tokenize a dataset for training or evaluation.
        
        Args:
            dataset: The dataset to tokenize
            text_column: Column containing the text (for unsupervised training)
            prompt_column: Column containing the prompt (for supervised training)
            completion_column: Column containing the completion (for supervised training)
            add_eos_token: Whether to add EOS token to the end of sequences
            
        Returns:
            Tokenized dataset
        """
        # Determine if we're doing supervised or unsupervised training
        is_supervised = prompt_column is not None and completion_column is not None
        
        def tokenize_function(examples):
            if is_supervised:
                # Supervised fine-tuning (instruction format)
                prompts = examples[prompt_column]
                completions = examples[completion_column]
                
                # Tokenize prompts and completions
                tokenized_prompts = self.tokenizer(
                    prompts,
                    truncation=True,
                    max_length=self.max_seq_length // 2,
                    padding=False,
                    return_tensors=None,
                )
                
                tokenized_completions = self.tokenizer(
                    completions,
                    truncation=True,
                    max_length=self.max_seq_length // 2,
                    padding=False,
                    return_tensors=None,
                )
                
                # Combine prompts and completions
                input_ids = []
                attention_mask = []
                labels = []
                
                for i in range(len(prompts)):
                    prompt_ids = tokenized_prompts["input_ids"][i]
                    completion_ids = tokenized_completions["input_ids"][i]
                    
                    # Add EOS token to completion if needed
                    if add_eos_token and completion_ids[-1] != self.tokenizer.eos_token_id:
                        completion_ids = completion_ids + [self.tokenizer.eos_token_id]
                    
                    # Combine prompt and completion
                    combined_ids = prompt_ids + completion_ids
                    
                    # Truncate if too long
                    if len(combined_ids) > self.max_seq_length:
                        combined_ids = combined_ids[:self.max_seq_length]
                    
                    # Create labels: -100 for prompt (ignored in loss), actual ids for completion
                    combined_labels = [-100] * len(prompt_ids) + completion_ids
                    
                    # Truncate labels if too long
                    if len(combined_labels) > self.max_seq_length:
                        combined_labels = combined_labels[:self.max_seq_length]
                    
                    # Create attention mask
                    combined_mask = [1] * len(combined_ids)
                    
                    input_ids.append(combined_ids)
                    attention_mask.append(combined_mask)
                    labels.append(combined_labels)
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            else:
                # Unsupervised training (causal language modeling)
                texts = examples[text_column]
                
                # Tokenize texts
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                    return_tensors=None,
                )
                
                # Add EOS token if needed
                if add_eos_token:
                    for i in range(len(tokenized["input_ids"])):
                        if tokenized["input_ids"][i][-1] != self.tokenizer.eos_token_id:
                            tokenized["input_ids"][i].append(self.tokenizer.eos_token_id)
                            tokenized["attention_mask"][i].append(1)
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=dataset.column_names,
        )
    
    def create_data_collator(self, pad_to_multiple_of: Optional[int] = 8):
        """
        Create a data collator for batching.
        
        Args:
            pad_to_multiple_of: Pad sequences to a multiple of this value
            
        Returns:
            Data collator function
        """
        def collate_fn(examples):
            # Get batch elements
            input_ids = [example["input_ids"] for example in examples]
            attention_mask = [example["attention_mask"] for example in examples]
            labels = [example["labels"] for example in examples]
            
            # Get max length in batch
            max_length = max(len(ids) for ids in input_ids)
            
            # Pad to multiple of pad_to_multiple_of if specified
            if pad_to_multiple_of is not None:
                max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            
            # Pad sequences
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            for ids, mask, lbl in zip(input_ids, attention_mask, labels):
                # Pad input_ids
                padding_length = max_length - len(ids)
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                padded_mask = mask + [0] * padding_length
                
                # Pad labels (with -100 to ignore in loss)
                padded_lbl = lbl + [-100] * padding_length
                
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
                padded_labels.append(padded_lbl)
            
            # Convert to tensors
            batch = {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
            }
            
            return batch
        
        return collate_fn
    
    def prepare_dataset_for_training(
        self,
        dataset_name: str,
        is_code_dataset: bool = True,
        is_multilingual: bool = False,
        train_test_split: bool = True,
        train_ratio: float = 0.9,
        **kwargs,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare a dataset for training.
        
        Args:
            dataset_name: Name of the dataset to prepare
            is_code_dataset: Whether this is a code dataset
            is_multilingual: Whether this is a multilingual dataset
            train_test_split: Whether to split into train/test
            train_ratio: Ratio of data to use for training
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Load the dataset
        dataset = self.load_dataset(dataset_name)
        
        # Split into train/test if needed
        if train_test_split:
            dataset = dataset.train_test_split(test_size=1-train_ratio, seed=42)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        # Preprocess based on dataset type
        if is_code_dataset:
            train_dataset = self.preprocess_code_dataset(train_dataset, **kwargs)
            if eval_dataset:
                eval_dataset = self.preprocess_code_dataset(eval_dataset, **kwargs)
        elif is_multilingual:
            train_dataset = self.preprocess_multilingual_dataset(train_dataset, **kwargs)
            if eval_dataset:
                eval_dataset = self.preprocess_multilingual_dataset(eval_dataset, **kwargs)
        
        # Tokenize datasets
        if is_code_dataset:
            train_dataset = self.tokenize_dataset(
                train_dataset,
                prompt_column="prompt",
                completion_column="completion",
            )
            if eval_dataset:
                eval_dataset = self.tokenize_dataset(
                    eval_dataset,
                    prompt_column="prompt",
                    completion_column="completion",
                )
        else:
            train_dataset = self.tokenize_dataset(train_dataset)
            if eval_dataset:
                eval_dataset = self.tokenize_dataset(eval_dataset)
        
        return train_dataset, eval_dataset
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        eval_batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create DataLoaders for training and evaluation.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            shuffle: Whether to shuffle the training data
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        # Create data collator
        collate_fn = self.create_data_collator()
        
        # Create training DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        # Create evaluation DataLoader if needed
        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size or batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        
        return train_dataloader, eval_dataloader


def load_and_process_dataset(config: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and process a dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer to use for processing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    data_config = config.get("data", {})
    
    # Initialize data processor
    processor = DataProcessor(
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 2048),
        data_dir=data_config.get("data_dir", "data"),
        preprocessing_num_workers=data_config.get("preprocessing_num_workers", 4),
    )
    
    # Prepare dataset
    dataset_name = data_config.get("dataset", "codealpaca")
    is_code_dataset = data_config.get("is_code_dataset", True)
    is_multilingual = data_config.get("is_multilingual", False)
    
    train_dataset, eval_dataset = processor.prepare_dataset_for_training(
        dataset_name=dataset_name,
        is_code_dataset=is_code_dataset,
        is_multilingual=is_multilingual,
        train_test_split=data_config.get("train_test_split", True),
        train_ratio=data_config.get("train_ratio", 0.9),
        instruction_column=data_config.get("instruction_column", "instruction"),
        input_column=data_config.get("input_column", "input"),
        output_column=data_config.get("output_column", "output"),
        prompt_template=data_config.get("prompt_template", None),
    )
    
    return train_dataset, eval_dataset
