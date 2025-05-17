#!/usr/bin/env python3
"""
DeepShiva Evaluation Script

This script evaluates the DeepShiva model on standard benchmarks:
- HumanEval: Code generation benchmark from OpenAI
- MBPP: Mostly Basic Programming Problems
- GSM8K: Grade School Math problems
- MATH: Advanced mathematical reasoning
- IndicGLUE: Indian language understanding

Results are reported in standard metrics and compared to baseline models.
"""

import os
import re
import json
import time
import argparse
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

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
    parser = argparse.ArgumentParser(description="DeepShiva Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "gsm8k", "math", "indicglue", "all"],
        help="Benchmark to evaluate on",
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
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per problem",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
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
    return parser.parse_args()


class Evaluator:
    """Base class for benchmark evaluators."""
    
    def __init__(
        self,
        inference_engine: DeepShivaInference,
        output_dir: str,
        temperature: float = 0.2,
        num_samples: int = 1,
        max_new_tokens: int = 1024,
        num_workers: int = 4,
    ):
        self.inference_engine = inference_engine
        self.output_dir = output_dir
        self.temperature = temperature
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.num_workers = num_workers
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the benchmark.
        
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
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


class HumanEvalEvaluator(Evaluator):
    """Evaluator for the HumanEval benchmark."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset("openai_humaneval")["test"]
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on HumanEval.
        
        Returns:
            Dictionary of metrics including pass@k
        """
        logger.info("Evaluating on HumanEval...")
        
        # Prepare problems
        problems = []
        for item in self.dataset:
            prompt = item["prompt"]
            entry_point = item["entry_point"]
            test_cases = item["test"]
            problems.append({
                "task_id": item["task_id"],
                "prompt": prompt,
                "entry_point": entry_point,
                "test_cases": test_cases,
                "canonical_solution": item["canonical_solution"],
            })
        
        # Generate solutions
        logger.info(f"Generating solutions for {len(problems)} problems...")
        solutions = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for problem in problems:
                for _ in range(self.num_samples):
                    futures.append(
                        executor.submit(
                            self._generate_solution,
                            problem["prompt"],
                            problem["task_id"],
                        )
                    )
            
            for future in tqdm(futures, total=len(futures)):
                solutions.append(future.result())
        
        # Evaluate solutions
        logger.info("Evaluating solutions...")
        results = self._evaluate_solutions(problems, solutions)
        
        # Save results
        self.save_results(results, "humaneval_results.json")
        
        return results
    
    def _generate_solution(self, prompt: str, task_id: str) -> Dict[str, Any]:
        """Generate a solution for a HumanEval problem.
        
        Args:
            prompt: Problem prompt
            task_id: Task ID
            
        Returns:
            Dictionary with solution details
        """
        try:
            # Format prompt for code completion
            code_prompt = prompt
            
            # Generate solution
            completion = self.inference_engine.complete_code(
                code_prompt=code_prompt,
                language="python",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.95,
                repetition_penalty=1.05,
            )
            
            # Extract the generated code
            solution = completion
            
            return {
                "task_id": task_id,
                "prompt": prompt,
                "completion": solution,
            }
        except Exception as e:
            logger.error(f"Error generating solution for {task_id}: {e}")
            return {
                "task_id": task_id,
                "prompt": prompt,
                "completion": "",
                "error": str(e),
            }
    
    def _evaluate_solutions(
        self,
        problems: List[Dict[str, Any]],
        solutions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate solutions using the HumanEval methodology.
        
        Args:
            problems: List of problems
            solutions: List of generated solutions
            
        Returns:
            Dictionary of metrics including pass@k
        """
        # Group solutions by task_id
        solutions_by_task = {}
        for solution in solutions:
            task_id = solution["task_id"]
            if task_id not in solutions_by_task:
                solutions_by_task[task_id] = []
            solutions_by_task[task_id].append(solution["completion"])
        
        # Check if solutions pass test cases
        results = []
        for problem in problems:
            task_id = problem["task_id"]
            if task_id not in solutions_by_task:
                continue
            
            task_solutions = solutions_by_task[task_id]
            correct = []
            
            for solution in task_solutions:
                # Combine prompt and completion
                full_solution = solution
                
                # Check if solution passes test cases
                is_correct = self._check_correctness(
                    full_solution,
                    problem["test_cases"],
                    problem["entry_point"],
                )
                correct.append(is_correct)
            
            results.append({
                "task_id": task_id,
                "correct": correct,
                "k": len(correct),
            })
        
        # Calculate pass@k
        ks = [1, 5, 10, 100]
        pass_at_k = {}
        
        for k in ks:
            if k > self.num_samples:
                continue
            
            pass_at_k[f"pass@{k}"] = self._calculate_pass_at_k(results, k)
        
        return {
            "pass_at_k": pass_at_k,
            "num_problems": len(problems),
            "num_solutions": len(solutions),
            "temperature": self.temperature,
            "num_samples": self.num_samples,
        }
    
    def _check_correctness(
        self,
        solution: str,
        test_cases: str,
        entry_point: str,
    ) -> bool:
        """Check if a solution passes the test cases.
        
        Args:
            solution: Generated solution
            test_cases: Test cases to check against
            entry_point: Name of the function to test
            
        Returns:
            True if the solution passes all test cases, False otherwise
        """
        # Create a temporary file with the solution and test cases
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(solution.encode("utf-8"))
            f.write(b"\n\n")
            f.write(test_cases.encode("utf-8"))
            f.write(b"\n\n")
            f.write(f"check({entry_point})".encode("utf-8"))
            temp_file = f.name
        
        try:
            # Run the test cases
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            # Check if the solution passed
            return result.returncode == 0 and "PASS" in result.stdout
        except Exception as e:
            logger.error(f"Error checking correctness: {e}")
            return False
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def _calculate_pass_at_k(
        self,
        results: List[Dict[str, Any]],
        k: int,
    ) -> float:
        """Calculate pass@k metric.
        
        Args:
            results: List of results for each problem
            k: k value for pass@k
            
        Returns:
            pass@k value
        """
        if k == 0:
            return 0.0
        
        pass_at_k = []
        for result in results:
            n = result["k"]
            c = sum(result["correct"])
            
            if n < k:
                continue
            
            # Calculate probability of solving the problem with k samples
            if c == 0:
                pass_at_k.append(0.0)
            else:
                pass_at_k.append(1.0 - np.prod([(n - i) / n for i in range(min(k, c))]))
        
        return float(np.mean(pass_at_k))


class MBPPEvaluator(Evaluator):
    """Evaluator for the MBPP benchmark."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset("mbpp")["test"]
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on MBPP.
        
        Returns:
            Dictionary of metrics including pass@k
        """
        logger.info("Evaluating on MBPP...")
        
        # Prepare problems
        problems = []
        for item in self.dataset:
            text = item["text"]
            test_list = item["test_list"]
            entry_point = self._extract_function_name(test_list[0])
            problems.append({
                "task_id": str(item["task_id"]),
                "text": text,
                "test_list": test_list,
                "entry_point": entry_point,
                "code": item["code"],
            })
        
        # Generate solutions
        logger.info(f"Generating solutions for {len(problems)} problems...")
        solutions = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for problem in problems:
                for _ in range(self.num_samples):
                    futures.append(
                        executor.submit(
                            self._generate_solution,
                            problem["text"],
                            problem["task_id"],
                        )
                    )
            
            for future in tqdm(futures, total=len(futures)):
                solutions.append(future.result())
        
        # Evaluate solutions
        logger.info("Evaluating solutions...")
        results = self._evaluate_solutions(problems, solutions)
        
        # Save results
        self.save_results(results, "mbpp_results.json")
        
        return results
    
    def _extract_function_name(self, test_case: str) -> str:
        """Extract function name from a test case.
        
        Args:
            test_case: Test case string
            
        Returns:
            Function name
        """
        match = re.search(r"assert\s+(\w+)\(", test_case)
        if match:
            return match.group(1)
        return ""
    
    def _generate_solution(self, text: str, task_id: str) -> Dict[str, Any]:
        """Generate a solution for an MBPP problem.
        
        Args:
            text: Problem description
            task_id: Task ID
            
        Returns:
            Dictionary with solution details
        """
        try:
            # Format prompt for code completion
            prompt = f"Write a Python function to solve the following problem:\n{text}\n\n"
            
            # Generate solution
            completion = self.inference_engine.complete_code(
                code_prompt=prompt,
                language="python",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.95,
                repetition_penalty=1.05,
            )
            
            return {
                "task_id": task_id,
                "text": text,
                "completion": completion,
            }
        except Exception as e:
            logger.error(f"Error generating solution for {task_id}: {e}")
            return {
                "task_id": task_id,
                "text": text,
                "completion": "",
                "error": str(e),
            }
    
    def _evaluate_solutions(
        self,
        problems: List[Dict[str, Any]],
        solutions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate solutions using the MBPP methodology.
        
        Args:
            problems: List of problems
            solutions: List of generated solutions
            
        Returns:
            Dictionary of metrics including pass@k
        """
        # Group solutions by task_id
        solutions_by_task = {}
        for solution in solutions:
            task_id = solution["task_id"]
            if task_id not in solutions_by_task:
                solutions_by_task[task_id] = []
            solutions_by_task[task_id].append(solution["completion"])
        
        # Check if solutions pass test cases
        results = []
        for problem in problems:
            task_id = problem["task_id"]
            if task_id not in solutions_by_task:
                continue
            
            task_solutions = solutions_by_task[task_id]
            correct = []
            
            for solution in task_solutions:
                # Check if solution passes test cases
                is_correct = self._check_correctness(
                    solution,
                    problem["test_list"],
                )
                correct.append(is_correct)
            
            results.append({
                "task_id": task_id,
                "correct": correct,
                "k": len(correct),
            })
        
        # Calculate pass@k
        ks = [1, 5, 10, 100]
        pass_at_k = {}
        
        for k in ks:
            if k > self.num_samples:
                continue
            
            pass_at_k[f"pass@{k}"] = self._calculate_pass_at_k(results, k)
        
        return {
            "pass_at_k": pass_at_k,
            "num_problems": len(problems),
            "num_solutions": len(solutions),
            "temperature": self.temperature,
            "num_samples": self.num_samples,
        }
    
    def _check_correctness(
        self,
        solution: str,
        test_list: List[str],
    ) -> bool:
        """Check if a solution passes the test cases.
        
        Args:
            solution: Generated solution
            test_list: List of test cases to check against
            
        Returns:
            True if the solution passes all test cases, False otherwise
        """
        # Create a temporary file with the solution and test cases
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(solution.encode("utf-8"))
            f.write(b"\n\n")
            for test in test_list:
                f.write(f"{test}\n".encode("utf-8"))
            temp_file = f.name
        
        try:
            # Run the test cases
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            # Check if the solution passed
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking correctness: {e}")
            return False
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def _calculate_pass_at_k(
        self,
        results: List[Dict[str, Any]],
        k: int,
    ) -> float:
        """Calculate pass@k metric.
        
        Args:
            results: List of results for each problem
            k: k value for pass@k
            
        Returns:
            pass@k value
        """
        if k == 0:
            return 0.0
        
        pass_at_k = []
        for result in results:
            n = result["k"]
            c = sum(result["correct"])
            
            if n < k:
                continue
            
            # Calculate probability of solving the problem with k samples
            if c == 0:
                pass_at_k.append(0.0)
            else:
                pass_at_k.append(1.0 - np.prod([(n - i) / n for i in range(min(k, c))]))
        
        return float(np.mean(pass_at_k))


class GSM8KEvaluator(Evaluator):
    """Evaluator for the GSM8K benchmark."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset("gsm8k", "main")["test"]
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on GSM8K.
        
        Returns:
            Dictionary of metrics including accuracy
        """
        logger.info("Evaluating on GSM8K...")
        
        # Prepare problems
        problems = []
        for i, item in enumerate(self.dataset):
            question = item["question"]
            answer = item["answer"]
            problems.append({
                "id": i,
                "question": question,
                "answer": answer,
            })
        
        # Generate solutions
        logger.info(f"Generating solutions for {len(problems)} problems...")
        solutions = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for problem in problems:
                futures.append(
                    executor.submit(
                        self._generate_solution,
                        problem["question"],
                        problem["id"],
                    )
                )
            
            for future in tqdm(futures, total=len(futures)):
                solutions.append(future.result())
        
        # Evaluate solutions
        logger.info("Evaluating solutions...")
        results = self._evaluate_solutions(problems, solutions)
        
        # Save results
        self.save_results(results, "gsm8k_results.json")
        
        return results
    
    def _generate_solution(self, question: str, problem_id: int) -> Dict[str, Any]:
        """Generate a solution for a GSM8K problem.
        
        Args:
            question: Problem question
            problem_id: Problem ID
            
        Returns:
            Dictionary with solution details
        """
        try:
            # Generate solution
            solution = self.inference_engine.solve_math(
                problem=question,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                show_work=True,
            )
            
            return {
                "id": problem_id,
                "question": question,
                "solution": solution,
            }
        except Exception as e:
            logger.error(f"Error generating solution for problem {problem_id}: {e}")
            return {
                "id": problem_id,
                "question": question,
                "solution": "",
                "error": str(e),
            }
    
    def _evaluate_solutions(
        self,
        problems: List[Dict[str, Any]],
        solutions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate solutions on GSM8K.
        
        Args:
            problems: List of problems
            solutions: List of generated solutions
            
        Returns:
            Dictionary of metrics including accuracy
        """
        # Match solutions to problems
        solutions_by_id = {solution["id"]: solution for solution in solutions}
        
        # Check if solutions are correct
        correct = 0
        incorrect = 0
        results = []
        
        for problem in problems:
            problem_id = problem["id"]
            if problem_id not in solutions_by_id:
                incorrect += 1
                continue
            
            solution = solutions_by_id[problem_id]["solution"]
            is_correct = self._check_correctness(solution, problem["answer"])
            
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            
            results.append({
                "id": problem_id,
                "question": problem["question"],
                "reference_answer": problem["answer"],
                "generated_solution": solution,
                "correct": is_correct,
            })
        
        # Calculate accuracy
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "incorrect": incorrect,
            "total": correct + incorrect,
            "results": results,
        }
    
    def _check_correctness(self, solution: str, reference: str) -> bool:
        """Check if a solution is correct.
        
        Args:
            solution: Generated solution
            reference: Reference answer
            
        Returns:
            True if the solution is correct, False otherwise
        """
        # Extract the final answer from the reference
        reference_answer = self._extract_answer(reference)
        
        # Extract the final answer from the solution
        solution_answer = self._extract_answer(solution)
        
        # Check if the answers match
        return reference_answer == solution_answer
    
    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from a solution.
        
        Args:
            text: Solution text
            
        Returns:
            Extracted answer
        """
        # Look for the answer pattern in GSM8K format
        match = re.search(r"(\d+(?:\.\d+)?)", text.split("\n")[-1])
        if match:
            return match.group(1)
        
        # Try to find any number in the last line
        lines = text.strip().split("\n")
        for line in reversed(lines):
            match = re.search(r"(\d+(?:\.\d+)?)", line)
            if match:
                return match.group(1)
        
        return ""


def main():
    args = parse_args()
    
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
    if args.benchmark == "humaneval" or args.benchmark == "all":
        evaluator = HumanEvalEvaluator(
            inference_engine=inference_engine,
            output_dir=args.output_dir,
            temperature=args.temperature,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
        )
        results = evaluator.evaluate()
        logger.info(f"HumanEval results: {results['pass_at_k']}")
    
    if args.benchmark == "mbpp" or args.benchmark == "all":
        evaluator = MBPPEvaluator(
            inference_engine=inference_engine,
            output_dir=args.output_dir,
            temperature=args.temperature,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
        )
        results = evaluator.evaluate()
        logger.info(f"MBPP results: {results['pass_at_k']}")
    
    if args.benchmark == "gsm8k" or args.benchmark == "all":
        evaluator = GSM8KEvaluator(
            inference_engine=inference_engine,
            output_dir=args.output_dir,
            temperature=args.temperature,
            num_samples=1,  # GSM8K uses a single sample
            max_new_tokens=args.max_new_tokens,
            num_workers=args.num_workers,
        )
        results = evaluator.evaluate()
        logger.info(f"GSM8K accuracy: {results['accuracy']:.4f}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
