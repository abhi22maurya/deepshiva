#!/usr/bin/env python3
"""
DeepShiva Evaluation Report Generator

This script generates a comprehensive markdown report from evaluation results:
- Summarizes performance across all benchmarks
- Compares DeepShiva with other state-of-the-art models
- Includes visualizations and detailed metrics
- Exports to markdown for easy conversion to PDF or HTML
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShiva Evaluation Report Generator")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--visualizations_dir",
        type=str,
        default="evaluation/visualizations",
        help="Directory containing visualizations",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation/DeepShiva_Evaluation_Report.md",
        help="Output markdown file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model used for evaluation",
    )
    parser.add_argument(
        "--include_comparison",
        action="store_true",
        help="Include comparison with other models",
    )
    return parser.parse_args()


class ReportGenerator:
    """Generates comprehensive evaluation reports for DeepShiva."""
    
    # Baseline model performance for comparison
    # These values are representative and should be updated with actual benchmarks
    BASELINE_MODELS = {
        "humaneval": {
            "pass@1": {
                "DeepSeek-Coder-V2-33B": 0.756,
                "CodeLlama-34B": 0.676,
                "GPT-4": 0.670,
                "Claude-2": 0.561,
                "Gemini-Pro": 0.689,
            },
            "pass@10": {
                "DeepSeek-Coder-V2-33B": 0.896,
                "CodeLlama-34B": 0.856,
                "GPT-4": 0.844,
                "Claude-2": 0.739,
                "Gemini-Pro": 0.851,
            },
        },
        "mbpp": {
            "pass@1": {
                "DeepSeek-Coder-V2-33B": 0.701,
                "CodeLlama-34B": 0.648,
                "GPT-4": 0.665,
                "Claude-2": 0.591,
                "Gemini-Pro": 0.656,
            },
            "pass@10": {
                "DeepSeek-Coder-V2-33B": 0.862,
                "CodeLlama-34B": 0.819,
                "GPT-4": 0.831,
                "Claude-2": 0.763,
                "Gemini-Pro": 0.825,
            },
        },
        "gsm8k": {
            "accuracy": {
                "DeepSeek-Coder-V2-33B": 0.571,
                "CodeLlama-34B": 0.429,
                "GPT-4": 0.921,
                "Claude-2": 0.878,
                "Gemini-Pro": 0.894,
            },
        },
    }
    
    def __init__(
        self,
        results_dir: str,
        visualizations_dir: str,
        output_file: str,
        model_path: Optional[str] = None,
        include_comparison: bool = True,
    ):
        self.results_dir = results_dir
        self.visualizations_dir = visualizations_dir
        self.output_file = output_file
        self.model_path = model_path
        self.include_comparison = include_comparison
        
        # Load results
        self.results = self._load_results()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    def _load_results(self) -> Dict[str, Any]:
        """Load evaluation results from files.
        
        Returns:
            Dictionary of results by benchmark
        """
        results = {}
        
        # Check for HumanEval results
        humaneval_path = os.path.join(self.results_dir, "humaneval_results.json")
        if os.path.exists(humaneval_path):
            with open(humaneval_path, "r", encoding="utf-8") as f:
                results["humaneval"] = json.load(f)
        
        # Check for MBPP results
        mbpp_path = os.path.join(self.results_dir, "mbpp_results.json")
        if os.path.exists(mbpp_path):
            with open(mbpp_path, "r", encoding="utf-8") as f:
                results["mbpp"] = json.load(f)
        
        # Check for GSM8K results
        gsm8k_path = os.path.join(self.results_dir, "gsm8k_results.json")
        if os.path.exists(gsm8k_path):
            with open(gsm8k_path, "r", encoding="utf-8") as f:
                results["gsm8k"] = json.load(f)
        
        return results
    
    def _get_visualization_path(self, filename: str) -> str:
        """Get the relative path to a visualization file.
        
        Args:
            filename: Visualization filename
            
        Returns:
            Relative path to the visualization
        """
        # Get the relative path from the output file to the visualization
        output_dir = os.path.dirname(self.output_file)
        rel_path = os.path.relpath(self.visualizations_dir, output_dir)
        return os.path.join(rel_path, filename)
    
    def generate_report(self) -> None:
        """Generate the evaluation report."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            # Write report header
            f.write("# DeepShiva Evaluation Report\n\n")
            
            # Add date and model information
            f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n")
            if self.model_path:
                f.write(f"**Model:** {self.model_path}\n\n")
            
            # Write executive summary
            f.write("## Executive Summary\n\n")
            f.write("DeepShiva is an open-source Mixture-of-Experts (MoE) code language model designed for coding tasks, mathematical reasoning, and multilingual support. This report presents the evaluation results on standard benchmarks and compares DeepShiva's performance with other state-of-the-art models.\n\n")
            
            # Add summary table
            f.write("### Performance Summary\n\n")
            summary_table = self._generate_summary_table()
            f.write(summary_table)
            f.write("\n\n")
            
            # Add overall comparison if available
            if self.include_comparison and os.path.exists(os.path.join(self.visualizations_dir, "overall_comparison.png")):
                f.write("### Overall Comparison\n\n")
                f.write(f"![Overall Comparison]({self._get_visualization_path('overall_comparison.png')})\n\n")
                f.write("This chart compares DeepShiva with other state-of-the-art models across different benchmarks.\n\n")
            
            # Write detailed results for each benchmark
            if "humaneval" in self.results:
                f.write(self._generate_humaneval_section())
            
            if "mbpp" in self.results:
                f.write(self._generate_mbpp_section())
            
            if "gsm8k" in self.results:
                f.write(self._generate_gsm8k_section())
            
            # Write methodology section
            f.write("## Evaluation Methodology\n\n")
            f.write("### HumanEval\n\n")
            f.write("The HumanEval benchmark consists of 164 hand-written programming problems, each with a function signature, docstring, body, and several unit tests. The model is evaluated on its ability to generate correct function implementations based on the signature and docstring. Performance is measured using the pass@k metric, which estimates the probability that the model would generate a correct solution if allowed k sampling attempts.\n\n")
            
            f.write("### MBPP\n\n")
            f.write("The Mostly Basic Programming Problems (MBPP) benchmark consists of 974 programming problems designed to be solvable by entry-level programmers. Each problem includes a task description, code solution, and test cases. Similar to HumanEval, performance is measured using the pass@k metric.\n\n")
            
            f.write("### GSM8K\n\n")
            f.write("The Grade School Math 8K (GSM8K) benchmark consists of 8,500 grade school math problems. The model is evaluated on its ability to solve these problems, which requires mathematical reasoning. Performance is measured using accuracy, which is the percentage of problems solved correctly.\n\n")
            
            # Write conclusion
            f.write("## Conclusion\n\n")
            f.write("Based on the evaluation results, DeepShiva demonstrates competitive performance across coding and mathematical reasoning tasks. ")
            
            # Add benchmark-specific conclusions
            conclusions = []
            if "humaneval" in self.results:
                humaneval_pass_at_1 = self.results["humaneval"]["pass_at_k"].get("pass@1", 0)
                if humaneval_pass_at_1 > 0.65:
                    conclusions.append("It shows strong performance on the HumanEval benchmark, indicating good code generation capabilities")
                elif humaneval_pass_at_1 > 0.5:
                    conclusions.append("It shows competitive performance on the HumanEval benchmark")
                else:
                    conclusions.append("There is room for improvement on the HumanEval benchmark")
            
            if "mbpp" in self.results:
                mbpp_pass_at_1 = self.results["mbpp"]["pass_at_k"].get("pass@1", 0)
                if mbpp_pass_at_1 > 0.65:
                    conclusions.append("demonstrates strong performance on the MBPP benchmark, showing good problem-solving abilities for basic programming tasks")
                elif mbpp_pass_at_1 > 0.5:
                    conclusions.append("shows competitive performance on the MBPP benchmark")
                else:
                    conclusions.append("has potential for improvement on the MBPP benchmark")
            
            if "gsm8k" in self.results:
                gsm8k_accuracy = self.results["gsm8k"]["accuracy"]
                if gsm8k_accuracy > 0.7:
                    conclusions.append("excels at mathematical reasoning as evidenced by its high performance on the GSM8K benchmark")
                elif gsm8k_accuracy > 0.5:
                    conclusions.append("demonstrates good mathematical reasoning capabilities on the GSM8K benchmark")
                else:
                    conclusions.append("shows potential in mathematical reasoning with room for improvement on the GSM8K benchmark")
            
            if conclusions:
                f.write(", and ".join(conclusions) + ". ")
            
            f.write("Future work will focus on further improving the model's performance through additional training on diverse datasets and refinement of the Mixture-of-Experts architecture.\n\n")
            
            # Add appendix with detailed results
            f.write("## Appendix: Detailed Results\n\n")
            f.write("Detailed results for each benchmark are available in the `evaluation/results` directory.\n")
        
        print(f"Report generated at {self.output_file}")
    
    def _generate_summary_table(self) -> str:
        """Generate a summary table of results.
        
        Returns:
            Markdown table with summary results
        """
        headers = ["Benchmark", "Metric", "DeepShiva"]
        
        if self.include_comparison:
            comparison_models = ["DeepSeek-Coder-V2-33B", "CodeLlama-34B", "GPT-4"]
            headers.extend(comparison_models)
        
        rows = []
        
        # Add HumanEval results
        if "humaneval" in self.results:
            pass_at_1 = self.results["humaneval"]["pass_at_k"].get("pass@1", "N/A")
            row = ["HumanEval", "pass@1", f"{pass_at_1:.3f}" if isinstance(pass_at_1, float) else pass_at_1]
            
            if self.include_comparison:
                for model in comparison_models:
                    baseline = self.BASELINE_MODELS.get("humaneval", {}).get("pass@1", {}).get(model, "N/A")
                    row.append(f"{baseline:.3f}" if isinstance(baseline, float) else baseline)
            
            rows.append(row)
        
        # Add MBPP results
        if "mbpp" in self.results:
            pass_at_1 = self.results["mbpp"]["pass_at_k"].get("pass@1", "N/A")
            row = ["MBPP", "pass@1", f"{pass_at_1:.3f}" if isinstance(pass_at_1, float) else pass_at_1]
            
            if self.include_comparison:
                for model in comparison_models:
                    baseline = self.BASELINE_MODELS.get("mbpp", {}).get("pass@1", {}).get(model, "N/A")
                    row.append(f"{baseline:.3f}" if isinstance(baseline, float) else baseline)
            
            rows.append(row)
        
        # Add GSM8K results
        if "gsm8k" in self.results:
            accuracy = self.results["gsm8k"].get("accuracy", "N/A")
            row = ["GSM8K", "accuracy", f"{accuracy:.3f}" if isinstance(accuracy, float) else accuracy]
            
            if self.include_comparison:
                for model in comparison_models:
                    baseline = self.BASELINE_MODELS.get("gsm8k", {}).get("accuracy", {}).get(model, "N/A")
                    row.append(f"{baseline:.3f}" if isinstance(baseline, float) else baseline)
            
            rows.append(row)
        
        # Generate table
        return tabulate(rows, headers=headers, tablefmt="pipe")
    
    def _generate_humaneval_section(self) -> str:
        """Generate the HumanEval results section.
        
        Returns:
            Markdown content for the HumanEval section
        """
        content = "## HumanEval Results\n\n"
        content += "The HumanEval benchmark evaluates the model's ability to generate correct Python function implementations based on docstrings.\n\n"
        
        # Add pass@k results
        content += "### pass@k Results\n\n"
        
        pass_at_k = self.results["humaneval"]["pass_at_k"]
        pass_at_k_table = []
        
        for k, score in sorted(pass_at_k.items(), key=lambda x: int(x[0].split("@")[1])):
            pass_at_k_table.append([k, f"{score:.3f}"])
        
        content += tabulate(pass_at_k_table, headers=["Metric", "Score"], tablefmt="pipe")
        content += "\n\n"
        
        # Add visualization if available
        if os.path.exists(os.path.join(self.visualizations_dir, "humaneval_pass_at_k.png")):
            content += f"![HumanEval pass@k]({self._get_visualization_path('humaneval_pass_at_k.png')})\n\n"
        
        # Add comparison if available
        if self.include_comparison and os.path.exists(os.path.join(self.visualizations_dir, "humaneval_pass@1_comparison.png")):
            content += "### Comparison with Other Models\n\n"
            content += f"![HumanEval Comparison]({self._get_visualization_path('humaneval_pass@1_comparison.png')})\n\n"
            
            # Add comparison table
            if "pass@1" in pass_at_k:
                content += "#### pass@1 Comparison\n\n"
                
                comparison_table = []
                deepshiva_score = pass_at_k["pass@1"]
                comparison_table.append(["DeepShiva", f"{deepshiva_score:.3f}"])
                
                baseline_models = self.BASELINE_MODELS.get("humaneval", {}).get("pass@1", {})
                for model, score in sorted(baseline_models.items(), key=lambda x: x[1], reverse=True):
                    comparison_table.append([model, f"{score:.3f}"])
                
                content += tabulate(comparison_table, headers=["Model", "pass@1"], tablefmt="pipe")
                content += "\n\n"
        
        return content
    
    def _generate_mbpp_section(self) -> str:
        """Generate the MBPP results section.
        
        Returns:
            Markdown content for the MBPP section
        """
        content = "## MBPP Results\n\n"
        content += "The Mostly Basic Programming Problems (MBPP) benchmark evaluates the model's ability to solve basic programming problems.\n\n"
        
        # Add pass@k results
        content += "### pass@k Results\n\n"
        
        pass_at_k = self.results["mbpp"]["pass_at_k"]
        pass_at_k_table = []
        
        for k, score in sorted(pass_at_k.items(), key=lambda x: int(x[0].split("@")[1])):
            pass_at_k_table.append([k, f"{score:.3f}"])
        
        content += tabulate(pass_at_k_table, headers=["Metric", "Score"], tablefmt="pipe")
        content += "\n\n"
        
        # Add visualization if available
        if os.path.exists(os.path.join(self.visualizations_dir, "mbpp_pass_at_k.png")):
            content += f"![MBPP pass@k]({self._get_visualization_path('mbpp_pass_at_k.png')})\n\n"
        
        # Add comparison if available
        if self.include_comparison and os.path.exists(os.path.join(self.visualizations_dir, "mbpp_pass@1_comparison.png")):
            content += "### Comparison with Other Models\n\n"
            content += f"![MBPP Comparison]({self._get_visualization_path('mbpp_pass@1_comparison.png')})\n\n"
            
            # Add comparison table
            if "pass@1" in pass_at_k:
                content += "#### pass@1 Comparison\n\n"
                
                comparison_table = []
                deepshiva_score = pass_at_k["pass@1"]
                comparison_table.append(["DeepShiva", f"{deepshiva_score:.3f}"])
                
                baseline_models = self.BASELINE_MODELS.get("mbpp", {}).get("pass@1", {})
                for model, score in sorted(baseline_models.items(), key=lambda x: x[1], reverse=True):
                    comparison_table.append([model, f"{score:.3f}"])
                
                content += tabulate(comparison_table, headers=["Model", "pass@1"], tablefmt="pipe")
                content += "\n\n"
        
        return content
    
    def _generate_gsm8k_section(self) -> str:
        """Generate the GSM8K results section.
        
        Returns:
            Markdown content for the GSM8K section
        """
        content = "## GSM8K Results\n\n"
        content += "The Grade School Math 8K (GSM8K) benchmark evaluates the model's mathematical reasoning capabilities.\n\n"
        
        # Add accuracy results
        content += "### Accuracy\n\n"
        
        accuracy = self.results["gsm8k"]["accuracy"]
        correct = self.results["gsm8k"]["correct"]
        total = self.results["gsm8k"]["total"]
        
        content += f"- **Accuracy:** {accuracy:.3f}\n"
        content += f"- **Correct:** {correct}\n"
        content += f"- **Total:** {total}\n\n"
        
        # Add visualization if available
        if os.path.exists(os.path.join(self.visualizations_dir, "gsm8k_accuracy.png")):
            content += f"![GSM8K Accuracy]({self._get_visualization_path('gsm8k_accuracy.png')})\n\n"
        
        # Add comparison if available
        if self.include_comparison and os.path.exists(os.path.join(self.visualizations_dir, "gsm8k_accuracy_comparison.png")):
            content += "### Comparison with Other Models\n\n"
            content += f"![GSM8K Comparison]({self._get_visualization_path('gsm8k_accuracy_comparison.png')})\n\n"
            
            # Add comparison table
            content += "#### Accuracy Comparison\n\n"
            
            comparison_table = []
            comparison_table.append(["DeepShiva", f"{accuracy:.3f}"])
            
            baseline_models = self.BASELINE_MODELS.get("gsm8k", {}).get("accuracy", {})
            for model, score in sorted(baseline_models.items(), key=lambda x: x[1], reverse=True):
                comparison_table.append([model, f"{score:.3f}"])
            
            content += tabulate(comparison_table, headers=["Model", "Accuracy"], tablefmt="pipe")
            content += "\n\n"
        
        # Add example problems and solutions
        if "results" in self.results["gsm8k"]:
            content += "### Example Problems and Solutions\n\n"
            
            # Get a few examples (correct and incorrect)
            examples = self.results["gsm8k"]["results"]
            correct_examples = [ex for ex in examples if ex["correct"]][:2]
            incorrect_examples = [ex for ex in examples if not ex["correct"]][:2]
            
            # Add correct examples
            if correct_examples:
                content += "#### Correctly Solved Examples\n\n"
                
                for i, example in enumerate(correct_examples):
                    content += f"**Example {i+1}:**\n\n"
                    content += f"**Question:** {example['question']}\n\n"
                    content += f"**Generated Solution:**\n```\n{example['generated_solution']}\n```\n\n"
                    content += f"**Reference Answer:** {example['reference_answer']}\n\n"
            
            # Add incorrect examples
            if incorrect_examples:
                content += "#### Incorrectly Solved Examples\n\n"
                
                for i, example in enumerate(incorrect_examples):
                    content += f"**Example {i+1}:**\n\n"
                    content += f"**Question:** {example['question']}\n\n"
                    content += f"**Generated Solution:**\n```\n{example['generated_solution']}\n```\n\n"
                    content += f"**Reference Answer:** {example['reference_answer']}\n\n"
        
        return content


def main():
    args = parse_args()
    
    # Create report generator
    generator = ReportGenerator(
        results_dir=args.results_dir,
        visualizations_dir=args.visualizations_dir,
        output_file=args.output_file,
        model_path=args.model_path,
        include_comparison=args.include_comparison,
    )
    
    # Generate report
    generator.generate_report()


if __name__ == "__main__":
    main()
