#!/usr/bin/env python3
"""
Dataset Downloader for DeepShiva

This script downloads and prepares datasets for training DeepShiva, including:
- Code datasets (HumanEval, MBPP, CodeAlpaca)
- Multilingual datasets (mC4, OSCAR, IndicCorp)
- Mathematical reasoning datasets (GSM8K, MATH)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for DeepShiva training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="codealpaca",
        choices=[
            "codealpaca", "humaneval", "mbpp", "gsm8k", "math",
            "mc4", "oscar", "indiccorp", "all_code", "all_math", "all_multilingual"
        ],
        help="Dataset to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save the downloaded datasets",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["hi", "ta", "bn", "te", "mr", "en"],
        help="Languages to download for multilingual datasets",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of examples to sample from each dataset (None for all)",
    )
    return parser.parse_args()


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def download_codealpaca(output_dir: str, sample_size: Optional[int] = None) -> None:
    """Download the CodeAlpaca dataset."""
    logger.info("Downloading CodeAlpaca dataset...")
    
    # Try multiple methods to download the dataset
    try:
        # Method 1: Using Hugging Face datasets library
        logger.info("Attempting to download CodeAlpaca-20k using Hugging Face datasets...")
        dataset = load_dataset("sahil2801/CodeAlpaca-20k")
        
        # Convert the dataset to a list of dictionaries
        data = [example for example in dataset["train"]]
        
    except Exception as e:
        logger.warning(f"Error using Hugging Face datasets: {e}")
        
        # Method 2: Direct download (fallback)
        try:
            logger.info("Attempting direct download...")
            # Try the correct URLs for this dataset
            urls = [
                "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json",
                "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/raw/main/code_alpaca_20k.json",
                "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json"
            ]
            
            success = False
            for url in urls:
                logger.info(f"Trying URL: {url}")
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    success = True
                    break
                else:
                    logger.warning(f"Failed with status code: {response.status_code}")
            
            if not success:
                logger.warning("All download attempts failed. Creating a small sample dataset instead...")
                # Create a small sample dataset
                data = [
                    {
                        "instruction": "Write a function to calculate factorial",
                        "input": "n = 5",
                        "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)"
                    },
                    {
                        "instruction": "Create a function to check if a number is prime",
                        "input": "17",
                        "output": "def is_prime(num):\n    if num <= 1:\n        return False\n    for i in range(2, int(num**0.5) + 1):\n        if num % i == 0:\n            return False\n    return True"
                    }
                ]
        
        except Exception as e:
            logger.error(f"Error with direct download: {e}")
            return
    
    # Sample if needed
    if sample_size is not None and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
    
    # Save the dataset
    output_path = os.path.join(output_dir, "codealpaca.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"CodeAlpaca dataset saved to {output_path} ({len(data)} examples)")


def download_humaneval(output_dir: str, sample_size: Optional[int] = None) -> None:
    """Download the HumanEval dataset."""
    logger.info("Downloading HumanEval dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("openai_humaneval")
    
    # Convert to the format we need
    data = []
    for item in dataset["test"]:
        entry = {
            "instruction": f"Write a Python function to solve the following problem:\n{item['prompt'].split('"""')[1].strip()}",
            "input": item['prompt'].split('"""')[0].strip() + "\n\n",
            "output": item["canonical_solution"],
        }
        data.append(entry)
    
    # Sample if needed
    if sample_size is not None and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
    
    # Save the dataset
    output_path = os.path.join(output_dir, "humaneval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"HumanEval dataset saved to {output_path} ({len(data)} examples)")


def download_mbpp(output_dir: str, sample_size: Optional[int] = None) -> None:
    """Download the MBPP dataset."""
    logger.info("Downloading MBPP dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("mbpp")
    
    # Convert to the format we need
    data = []
    for item in dataset["test"]:
        entry = {
            "instruction": f"Write a Python function to solve the following problem:\n{item['text']}",
            "input": "\n".join(item["test_list"]),
            "output": item["code"],
        }
        data.append(entry)
    
    # Sample if needed
    if sample_size is not None and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
    
    # Save the dataset
    output_path = os.path.join(output_dir, "mbpp.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"MBPP dataset saved to {output_path} ({len(data)} examples)")


def download_gsm8k(output_dir: str, sample_size: Optional[int] = None) -> None:
    """Download the GSM8K dataset."""
    logger.info("Downloading GSM8K dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("gsm8k", "main")
    
    # Convert to the format we need
    data = []
    for item in dataset["train"]:
        entry = {
            "instruction": "Solve the following grade school math problem, showing your work step-by-step:",
            "input": item["question"],
            "output": item["answer"],
        }
        data.append(entry)
    
    # Sample if needed
    if sample_size is not None and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
    
    # Save the dataset
    output_path = os.path.join(output_dir, "gsm8k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"GSM8K dataset saved to {output_path} ({len(data)} examples)")


def download_math(output_dir: str, sample_size: Optional[int] = None) -> None:
    """Download the MATH dataset."""
    logger.info("Downloading MATH dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("hendrycks/math")
    
    # Convert to the format we need
    data = []
    for item in dataset["train"]:
        entry = {
            "instruction": f"Solve the following {item['type']} problem, showing your work step-by-step:",
            "input": item["problem"],
            "output": item["solution"],
        }
        data.append(entry)
    
    # Sample if needed
    if sample_size is not None and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
    
    # Save the dataset
    output_path = os.path.join(output_dir, "math.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"MATH dataset saved to {output_path} ({len(data)} examples)")


def download_multilingual(output_dir: str, languages: List[str], dataset_name: str, sample_size: Optional[int] = None) -> None:
    """Download a multilingual dataset."""
    logger.info(f"Downloading {dataset_name} dataset for languages: {languages}...")
    
    all_data = []
    
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        
        try:
            if dataset_name == "mc4":
                # mC4 dataset
                dataset = load_dataset("mc4", lang, split="train", streaming=True)
                data_iter = iter(dataset)
                lang_data = []
                
                # Get sample_size examples or 1000 if None
                count = sample_size or 1000
                for _ in tqdm(range(count)):
                    try:
                        item = next(data_iter)
                        lang_data.append({
                            "text": item["text"],
                            "language": lang,
                        })
                    except StopIteration:
                        break
                
                all_data.extend(lang_data)
                
            elif dataset_name == "oscar":
                # OSCAR dataset
                dataset = load_dataset("oscar", f"unshuffled_deduplicated_{lang}", split="train", streaming=True)
                data_iter = iter(dataset)
                lang_data = []
                
                # Get sample_size examples or 1000 if None
                count = sample_size or 1000
                for _ in tqdm(range(count)):
                    try:
                        item = next(data_iter)
                        lang_data.append({
                            "text": item["text"],
                            "language": lang,
                        })
                    except StopIteration:
                        break
                
                all_data.extend(lang_data)
                
            elif dataset_name == "indiccorp":
                # IndicCorp dataset (for Indian languages)
                if lang in ["hi", "ta", "bn", "te", "mr"]:
                    dataset = load_dataset("ai4bharat/IndicCorp", lang, split="train", streaming=True)
                    data_iter = iter(dataset)
                    lang_data = []
                    
                    # Get sample_size examples or 1000 if None
                    count = sample_size or 1000
                    for _ in tqdm(range(count)):
                        try:
                            item = next(data_iter)
                            lang_data.append({
                                "text": item["text"],
                                "language": lang,
                            })
                        except StopIteration:
                            break
                    
                    all_data.extend(lang_data)
                else:
                    logger.warning(f"Language {lang} not available in IndicCorp, skipping...")
        
        except Exception as e:
            logger.error(f"Error processing language {lang}: {e}")
    
    # Save the dataset
    output_path = os.path.join(output_dir, f"{dataset_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{dataset_name} dataset saved to {output_path} ({len(all_data)} examples)")


def main():
    args = parse_args()
    
    # Ensure the output directory exists
    ensure_dir(args.output_dir)
    
    # Download the specified dataset
    if args.dataset == "codealpaca":
        download_codealpaca(args.output_dir, args.sample_size)
    elif args.dataset == "humaneval":
        download_humaneval(args.output_dir, args.sample_size)
    elif args.dataset == "mbpp":
        download_mbpp(args.output_dir, args.sample_size)
    elif args.dataset == "gsm8k":
        download_gsm8k(args.output_dir, args.sample_size)
    elif args.dataset == "math":
        download_math(args.output_dir, args.sample_size)
    elif args.dataset == "mc4":
        download_multilingual(args.output_dir, args.languages, "mc4", args.sample_size)
    elif args.dataset == "oscar":
        download_multilingual(args.output_dir, args.languages, "oscar", args.sample_size)
    elif args.dataset == "indiccorp":
        download_multilingual(args.output_dir, args.languages, "indiccorp", args.sample_size)
    elif args.dataset == "all_code":
        download_codealpaca(args.output_dir, args.sample_size)
        download_humaneval(args.output_dir, args.sample_size)
        download_mbpp(args.output_dir, args.sample_size)
    elif args.dataset == "all_math":
        download_gsm8k(args.output_dir, args.sample_size)
        download_math(args.output_dir, args.sample_size)
    elif args.dataset == "all_multilingual":
        download_multilingual(args.output_dir, args.languages, "mc4", args.sample_size)
        download_multilingual(args.output_dir, args.languages, "indiccorp", args.sample_size)
    
    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()