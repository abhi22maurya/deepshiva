#!/usr/bin/env python3
"""
DeepShiva Multilingual Evaluation Script

This script evaluates the DeepShiva model on multilingual tasks:
- Translation between languages (especially Indian languages)
- Code understanding and generation in multiple languages
- Multilingual question answering
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch
from sacrebleu import corpus_bleu

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import DeepShivaInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShiva Multilingual Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="translation",
        choices=["translation", "code", "qa", "all"],
        help="Multilingual task to evaluate",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="hi,bn,ta,te,ml,mr,gu,pa,en",
        help="Comma-separated list of language codes to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Precision to use for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel execution",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to evaluate per language pair",
    )
    return parser.parse_args()


class MultilingualEvaluator:
    """Base class for multilingual evaluators."""
    
    def __init__(
        self,
        inference_engine: DeepShivaInference,
        languages: List[str],
        output_dir: str,
        num_workers: int = 4,
        sample_size: int = 100,
    ):
        self.inference_engine = inference_engine
        self.languages = languages
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.sample_size = sample_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on multilingual tasks.
        
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Dictionary of metrics
            filename: Name of the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")


class TranslationEvaluator(MultilingualEvaluator):
    """Evaluator for multilingual translation tasks."""
    
    # Language names for better readability
    LANGUAGE_NAMES = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "kn": "Kannada",
        "or": "Odia",
        "as": "Assamese",
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self._load_translation_dataset()
    
    def _load_translation_dataset(self) -> Dict[str, Any]:
        """Load translation datasets for evaluation.
        
        Returns:
            Dictionary of datasets by language pair
        """
        datasets = {}
        
        # Try to load Samanantar dataset for Indian languages
        try:
            for src_lang in self.languages:
                for tgt_lang in self.languages:
                    if src_lang != tgt_lang:
                        lang_pair = f"{src_lang}-{tgt_lang}"
                        
                        # For Indian languages, use Samanantar if available
                        if src_lang == "en" or tgt_lang == "en":
                            try:
                                # Determine which language is not English
                                indic_lang = tgt_lang if src_lang == "en" else src_lang
                                
                                # Load Samanantar dataset
                                dataset = load_dataset(
                                    "ai4bharat/samanantar",
                                    f"{indic_lang}-en",
                                    split="validation",
                                )
                                
                                # Limit to sample size
                                dataset = dataset.select(range(min(self.sample_size, len(dataset))))
                                
                                # Store in the correct direction
                                if src_lang == "en":
                                    # en -> indic_lang
                                    datasets[lang_pair] = {
                                        "source": dataset["en"],
                                        "target": dataset[indic_lang],
                                    }
                                else:
                                    # indic_lang -> en
                                    datasets[lang_pair] = {
                                        "source": dataset[indic_lang],
                                        "target": dataset["en"],
                                    }
                                
                                logger.info(f"Loaded Samanantar dataset for {lang_pair}")
                                continue
                            except Exception as e:
                                logger.warning(f"Failed to load Samanantar for {lang_pair}: {e}")
                        
                        # Fallback to FLORES dataset
                        try:
                            dataset = load_dataset("facebook/flores", "dev", split="dev")
                            
                            # Map language codes to FLORES codes
                            flores_codes = {
                                "en": "eng_Latn",
                                "hi": "hin_Deva",
                                "bn": "ben_Beng",
                                "ta": "tam_Taml",
                                "te": "tel_Telu",
                                "ml": "mal_Mlym",
                                "mr": "mar_Deva",
                                "gu": "guj_Gujr",
                                "pa": "pan_Guru",
                                "kn": "kan_Knda",
                                "or": "ory_Orya",
                                "as": "asm_Beng",
                            }
                            
                            if src_lang in flores_codes and tgt_lang in flores_codes:
                                src_flores = flores_codes[src_lang]
                                tgt_flores = flores_codes[tgt_lang]
                                
                                # Limit to sample size
                                dataset = dataset.select(range(min(self.sample_size, len(dataset))))
                                
                                datasets[lang_pair] = {
                                    "source": dataset["sentence_" + src_flores],
                                    "target": dataset["sentence_" + tgt_flores],
                                }
                                
                                logger.info(f"Loaded FLORES dataset for {lang_pair}")
                            else:
                                logger.warning(f"No FLORES mapping for {lang_pair}")
                        except Exception as e:
                            logger.warning(f"Failed to load FLORES for {lang_pair}: {e}")
        except Exception as e:
            logger.error(f"Error loading translation datasets: {e}")
        
        return datasets
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on translation tasks.
        
        Returns:
            Dictionary of metrics including BLEU scores
        """
        logger.info("Evaluating on translation tasks...")
        
        results = {}
        all_results = []
        
        # Evaluate each language pair
        for lang_pair, dataset in self.dataset.items():
            logger.info(f"Evaluating translation for {lang_pair}")
            
            src_lang, tgt_lang = lang_pair.split("-")
            src_texts = dataset["source"]
            tgt_texts = dataset["target"]
            
            # Generate translations
            translations = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for src_text in src_texts:
                    futures.append(
                        executor.submit(
                            self._translate,
                            src_text,
                            src_lang,
                            tgt_lang,
                        )
                    )
                
                for future in tqdm(futures, total=len(futures)):
                    translations.append(future.result())
            
            # Calculate BLEU score
            bleu = corpus_bleu(translations, [tgt_texts]).score
            
            # Store results
            pair_results = {
                "language_pair": lang_pair,
                "source_language": self.LANGUAGE_NAMES.get(src_lang, src_lang),
                "target_language": self.LANGUAGE_NAMES.get(tgt_lang, tgt_lang),
                "bleu": bleu,
                "num_samples": len(src_texts),
                "examples": [
                    {
                        "source": src,
                        "reference": ref,
                        "translation": trans,
                    }
                    for src, ref, trans in zip(src_texts[:5], tgt_texts[:5], translations[:5])
                ],
            }
            
            results[lang_pair] = pair_results
            all_results.append(pair_results)
            
            logger.info(f"BLEU score for {lang_pair}: {bleu:.2f}")
        
        # Calculate average BLEU score
        if all_results:
            avg_bleu = np.mean([r["bleu"] for r in all_results])
            results["average_bleu"] = avg_bleu
            logger.info(f"Average BLEU score: {avg_bleu:.2f}")
        
        # Save results
        self.save_results(results, "translation_results.json")
        
        return results
    
    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from source language to target language.
        
        Args:
            text: Source text
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            translation = self.inference_engine.translate(
                text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
            )
            return translation
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return ""


class MultilingualCodeEvaluator(MultilingualEvaluator):
    """Evaluator for multilingual code tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self._load_code_dataset()
    
    def _load_code_dataset(self) -> Dict[str, Any]:
        """Load code datasets for evaluation.
        
        Returns:
            Dictionary of datasets by programming language
        """
        datasets = {}
        
        # Define programming languages to evaluate
        prog_languages = ["python", "java", "javascript", "cpp", "go", "rust"]
        
        # Try to load The Stack dataset
        try:
            for lang in prog_languages:
                try:
                    # Load The Stack dataset
                    dataset = load_dataset(
                        "bigcode/the-stack-smol",
                        data_dir=lang,
                        split="test",
                    )
                    
                    # Limit to sample size
                    dataset = dataset.select(range(min(self.sample_size, len(dataset))))
                    
                    # Extract problem descriptions and solutions
                    problems = []
                    for item in dataset:
                        # Extract a function or class definition
                        code = item["content"]
                        if code and len(code.strip()) > 0:
                            # Create a problem by taking the first few lines as a prompt
                            lines = code.strip().split("\n")
                            if len(lines) > 5:
                                prompt = "\n".join(lines[:5])
                                solution = code
                                problems.append({
                                    "prompt": prompt,
                                    "solution": solution,
                                })
                    
                    if problems:
                        datasets[lang] = problems[:self.sample_size]
                        logger.info(f"Loaded {len(datasets[lang])} {lang} code samples")
                except Exception as e:
                    logger.warning(f"Failed to load dataset for {lang}: {e}")
        except Exception as e:
            logger.error(f"Error loading code datasets: {e}")
        
        return datasets
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on multilingual code tasks.
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating on multilingual code tasks...")
        
        results = {}
        all_results = []
        
        # Evaluate each programming language
        for lang, problems in self.dataset.items():
            logger.info(f"Evaluating code completion for {lang}")
            
            # Generate code completions
            completions = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for problem in problems:
                    futures.append(
                        executor.submit(
                            self._complete_code,
                            problem["prompt"],
                            lang,
                        )
                    )
                
                for future in tqdm(futures, total=len(futures)):
                    completions.append(future.result())
            
            # Calculate similarity metrics
            similarities = []
            for completion, problem in zip(completions, problems):
                # Calculate character-level similarity
                similarity = self._calculate_similarity(completion, problem["solution"])
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # Store results
            lang_results = {
                "language": lang,
                "average_similarity": avg_similarity,
                "num_samples": len(problems),
                "examples": [
                    {
                        "prompt": problem["prompt"],
                        "completion": completion,
                        "reference": problem["solution"],
                        "similarity": similarity,
                    }
                    for problem, completion, similarity in zip(problems[:5], completions[:5], similarities[:5])
                ],
            }
            
            results[lang] = lang_results
            all_results.append(lang_results)
            
            logger.info(f"Average similarity for {lang}: {avg_similarity:.2f}")
        
        # Calculate average similarity
        if all_results:
            avg_similarity = np.mean([r["average_similarity"] for r in all_results])
            results["average_similarity"] = avg_similarity
            logger.info(f"Average similarity across languages: {avg_similarity:.2f}")
        
        # Save results
        self.save_results(results, "multilingual_code_results.json")
        
        return results
    
    def _complete_code(self, prompt: str, language: str) -> str:
        """Complete code based on a prompt.
        
        Args:
            prompt: Code prompt
            language: Programming language
            
        Returns:
            Completed code
        """
        try:
            completion = self.inference_engine.complete_code(
                code_prompt=prompt,
                language=language,
                max_new_tokens=256,
                temperature=0.2,
            )
            return completion
        except Exception as e:
            logger.error(f"Error completing code: {e}")
            return ""
    
    def _calculate_similarity(self, generated: str, reference: str) -> float:
        """Calculate similarity between generated and reference code.
        
        Args:
            generated: Generated code
            reference: Reference code
            
        Returns:
            Similarity score (0-1)
        """
        # Simple character-level similarity
        generated = generated.strip()
        reference = reference.strip()
        
        if not generated or not reference:
            return 0.0
        
        # Calculate Levenshtein distance
        import Levenshtein
        distance = Levenshtein.distance(generated, reference)
        max_len = max(len(generated), len(reference))
        
        # Convert to similarity (0-1)
        similarity = 1.0 - (distance / max_len)
        return similarity


class MultilingualQAEvaluator(MultilingualEvaluator):
    """Evaluator for multilingual question answering tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self._load_qa_dataset()
    
    def _load_qa_dataset(self) -> Dict[str, Any]:
        """Load QA datasets for evaluation.
        
        Returns:
            Dictionary of datasets by language
        """
        datasets = {}
        
        # Try to load MLQA dataset
        try:
            for lang in self.languages:
                try:
                    # Map language codes to MLQA codes
                    mlqa_codes = {
                        "en": "english",
                        "hi": "hindi",
                        "ar": "arabic",
                        "de": "german",
                        "es": "spanish",
                        "vi": "vietnamese",
                        "zh": "chinese",
                    }
                    
                    if lang in mlqa_codes:
                        # Load MLQA dataset
                        dataset = load_dataset(
                            "mlqa",
                            f"{mlqa_codes[lang]}-{mlqa_codes[lang]}",
                            split="test",
                        )
                        
                        # Limit to sample size
                        dataset = dataset.select(range(min(self.sample_size, len(dataset))))
                        
                        # Extract questions and answers
                        qa_pairs = []
                        for item in dataset:
                            context = item["context"]
                            question = item["question"]
                            answers = item["answers"]
                            
                            if context and question and answers:
                                qa_pairs.append({
                                    "context": context,
                                    "question": question,
                                    "answers": answers["text"],
                                })
                        
                        if qa_pairs:
                            datasets[lang] = qa_pairs
                            logger.info(f"Loaded {len(qa_pairs)} QA pairs for {lang}")
                    else:
                        logger.warning(f"No MLQA mapping for {lang}")
                except Exception as e:
                    logger.warning(f"Failed to load MLQA for {lang}: {e}")
                    
                    # Fallback to XQuAD dataset for some languages
                    try:
                        # Map language codes to XQuAD codes
                        xquad_codes = {
                            "en": "xquad.en",
                            "hi": "xquad.hi",
                            "ar": "xquad.ar",
                            "de": "xquad.de",
                            "es": "xquad.es",
                            "vi": "xquad.vi",
                            "zh": "xquad.zh",
                            "el": "xquad.el",
                            "ru": "xquad.ru",
                            "th": "xquad.th",
                            "tr": "xquad.tr",
                        }
                        
                        if lang in xquad_codes:
                            # Load XQuAD dataset
                            dataset = load_dataset(
                                "xquad",
                                xquad_codes[lang],
                                split="validation",
                            )
                            
                            # Limit to sample size
                            dataset = dataset.select(range(min(self.sample_size, len(dataset))))
                            
                            # Extract questions and answers
                            qa_pairs = []
                            for item in dataset:
                                context = item["context"]
                                question = item["question"]
                                answers = item["answers"]
                                
                                if context and question and answers:
                                    qa_pairs.append({
                                        "context": context,
                                        "question": question,
                                        "answers": answers["text"],
                                    })
                            
                            if qa_pairs:
                                datasets[lang] = qa_pairs
                                logger.info(f"Loaded {len(qa_pairs)} QA pairs for {lang} from XQuAD")
                        else:
                            logger.warning(f"No XQuAD mapping for {lang}")
                    except Exception as e:
                        logger.warning(f"Failed to load XQuAD for {lang}: {e}")
        except Exception as e:
            logger.error(f"Error loading QA datasets: {e}")
        
        return datasets
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on multilingual QA tasks.
        
        Returns:
            Dictionary of metrics including F1 and exact match scores
        """
        logger.info("Evaluating on multilingual QA tasks...")
        
        results = {}
        all_results = []
        
        # Evaluate each language
        for lang, qa_pairs in self.dataset.items():
            logger.info(f"Evaluating QA for {lang}")
            
            # Generate answers
            predictions = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for qa_pair in qa_pairs:
                    futures.append(
                        executor.submit(
                            self._answer_question,
                            qa_pair["context"],
                            qa_pair["question"],
                        )
                    )
                
                for future in tqdm(futures, total=len(futures)):
                    predictions.append(future.result())
            
            # Calculate metrics
            f1_scores = []
            exact_matches = []
            
            for prediction, qa_pair in zip(predictions, qa_pairs):
                references = qa_pair["answers"]
                
                # Calculate F1 and exact match
                f1 = self._calculate_f1(prediction, references)
                em = self._calculate_exact_match(prediction, references)
                
                f1_scores.append(f1)
                exact_matches.append(em)
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            avg_em = np.mean(exact_matches) if exact_matches else 0.0
            
            # Store results
            lang_results = {
                "language": lang,
                "average_f1": avg_f1,
                "average_exact_match": avg_em,
                "num_samples": len(qa_pairs),
                "examples": [
                    {
                        "context": qa_pair["context"][:200] + "...",  # Truncate for readability
                        "question": qa_pair["question"],
                        "prediction": prediction,
                        "references": qa_pair["answers"],
                        "f1": f1,
                        "exact_match": em,
                    }
                    for qa_pair, prediction, f1, em in zip(
                        qa_pairs[:5], predictions[:5], f1_scores[:5], exact_matches[:5]
                    )
                ],
            }
            
            results[lang] = lang_results
            all_results.append(lang_results)
            
            logger.info(f"Average F1 for {lang}: {avg_f1:.2f}")
            logger.info(f"Average EM for {lang}: {avg_em:.2f}")
        
        # Calculate average metrics
        if all_results:
            avg_f1 = np.mean([r["average_f1"] for r in all_results])
            avg_em = np.mean([r["average_exact_match"] for r in all_results])
            
            results["average_f1"] = avg_f1
            results["average_exact_match"] = avg_em
            
            logger.info(f"Average F1 across languages: {avg_f1:.2f}")
            logger.info(f"Average EM across languages: {avg_em:.2f}")
        
        # Save results
        self.save_results(results, "multilingual_qa_results.json")
        
        return results
    
    def _answer_question(self, context: str, question: str) -> str:
        """Answer a question based on context.
        
        Args:
            context: Context text
            question: Question text
            
        Returns:
            Answer text
        """
        try:
            # Format prompt for QA
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate answer
            answer = self.inference_engine.generate_text(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.2,
            )
            
            return answer.strip()
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return ""
    
    def _calculate_f1(self, prediction: str, references: List[str]) -> float:
        """Calculate F1 score between prediction and references.
        
        Args:
            prediction: Predicted answer
            references: Reference answers
            
        Returns:
            F1 score (0-1)
        """
        if not prediction or not references:
            return 0.0
        
        # Normalize
        prediction = self._normalize_text(prediction)
        references = [self._normalize_text(ref) for ref in references]
        
        # Calculate F1 for each reference and take the maximum
        f1_scores = []
        
        for reference in references:
            # Tokenize
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
            
            # Calculate precision and recall
            common = set(pred_tokens) & set(ref_tokens)
            
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            
            # Calculate F1
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0
    
    def _calculate_exact_match(self, prediction: str, references: List[str]) -> float:
        """Calculate exact match between prediction and references.
        
        Args:
            prediction: Predicted answer
            references: Reference answers
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if not prediction or not references:
            return 0.0
        
        # Normalize
        prediction = self._normalize_text(prediction)
        references = [self._normalize_text(ref) for ref in references]
        
        # Check for exact match
        return float(any(prediction == ref for ref in references))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra whitespace
        import re
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


def main():
    args = parse_args()
    
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    logger.info(f"Loading model from {args.model_path}")
    inference_engine = DeepShivaInference(
        model_path=args.model_path,
        device=args.device,
        precision=args.precision,
    )
    logger.info("Model loaded successfully")
    
    # Run evaluations
    if args.task == "translation" or args.task == "all":
        evaluator = TranslationEvaluator(
            inference_engine=inference_engine,
            languages=languages,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            sample_size=args.sample_size,
        )
        results = evaluator.evaluate()
        if "average_bleu" in results:
            logger.info(f"Average BLEU score: {results['average_bleu']:.2f}")
    
    if args.task == "code" or args.task == "all":
        evaluator = MultilingualCodeEvaluator(
            inference_engine=inference_engine,
            languages=languages,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            sample_size=args.sample_size,
        )
        results = evaluator.evaluate()
        if "average_similarity" in results:
            logger.info(f"Average code similarity: {results['average_similarity']:.2f}")
    
    if args.task == "qa" or args.task == "all":
        evaluator = MultilingualQAEvaluator(
            inference_engine=inference_engine,
            languages=languages,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            sample_size=args.sample_size,
        )
        results = evaluator.evaluate()
        if "average_f1" in results:
            logger.info(f"Average F1 score: {results['average_f1']:.2f}")
        if "average_exact_match" in results:
            logger.info(f"Average exact match: {results['average_exact_match']:.2f}")
    
    logger.info("Multilingual evaluation complete!")


if __name__ == "__main__":
    main()
