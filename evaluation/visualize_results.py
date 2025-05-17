#!/usr/bin/env python3
"""
DeepShiva Evaluation Results Visualization

This script visualizes the evaluation results from the DeepShiva model benchmarks:
- Generates charts comparing DeepShiva to other models
- Creates detailed performance breakdowns
- Exports visualizations for reports and presentations
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configure visualization style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShiva Evaluation Results Visualization")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for visualizations",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Include comparison with other models",
    )
    return parser.parse_args()


class ResultsVisualizer:
    """Visualizes evaluation results for DeepShiva."""
    
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
    
    # Color scheme for DeepShiva and baseline models
    COLORS = {
        "DeepShiva": "#FF5733",  # Orange-red
        "DeepSeek-Coder-V2-33B": "#3366FF",  # Blue
        "CodeLlama-34B": "#33FF57",  # Green
        "GPT-4": "#9933FF",  # Purple
        "Claude-2": "#FF33E6",  # Pink
        "Gemini-Pro": "#33FFF5",  # Cyan
    }
    
    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        file_format: str = "png",
        include_comparison: bool = True,
    ):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.file_format = file_format
        self.include_comparison = include_comparison
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
    
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
    
    def visualize_all(self) -> None:
        """Generate all visualizations."""
        if "humaneval" in self.results:
            self.visualize_humaneval()
        
        if "mbpp" in self.results:
            self.visualize_mbpp()
        
        if "gsm8k" in self.results:
            self.visualize_gsm8k()
        
        if self.include_comparison and len(self.results) > 0:
            self.visualize_model_comparison()
    
    def visualize_humaneval(self) -> None:
        """Visualize HumanEval results."""
        if "humaneval" not in self.results:
            return
        
        results = self.results["humaneval"]
        pass_at_k = results["pass_at_k"]
        
        # Create bar chart for pass@k
        plt.figure(figsize=(10, 6))
        
        # Extract k values and scores
        k_values = [int(k.split("@")[1]) for k in pass_at_k.keys()]
        scores = list(pass_at_k.values())
        
        # Sort by k
        k_scores = sorted(zip(k_values, scores), key=lambda x: x[0])
        k_values = [k for k, _ in k_scores]
        scores = [s for _, s in k_scores]
        
        # Plot
        bars = plt.bar(
            [f"pass@{k}" for k in k_values],
            scores,
            color=self.COLORS["DeepShiva"],
            alpha=0.8,
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        plt.ylim(0, 1.0)
        plt.ylabel("Score")
        plt.title("DeepShiva Performance on HumanEval")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"humaneval_pass_at_k.{self.file_format}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        
        # Create comparison chart if requested
        if self.include_comparison and "pass@1" in pass_at_k:
            self._create_comparison_chart(
                benchmark="humaneval",
                metric="pass@1",
                deepshiva_value=pass_at_k["pass@1"],
                title="HumanEval pass@1 Comparison",
                ylabel="pass@1 Score",
            )
    
    def visualize_mbpp(self) -> None:
        """Visualize MBPP results."""
        if "mbpp" not in self.results:
            return
        
        results = self.results["mbpp"]
        pass_at_k = results["pass_at_k"]
        
        # Create bar chart for pass@k
        plt.figure(figsize=(10, 6))
        
        # Extract k values and scores
        k_values = [int(k.split("@")[1]) for k in pass_at_k.keys()]
        scores = list(pass_at_k.values())
        
        # Sort by k
        k_scores = sorted(zip(k_values, scores), key=lambda x: x[0])
        k_values = [k for k, _ in k_scores]
        scores = [s for _, s in k_scores]
        
        # Plot
        bars = plt.bar(
            [f"pass@{k}" for k in k_values],
            scores,
            color=self.COLORS["DeepShiva"],
            alpha=0.8,
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        plt.ylim(0, 1.0)
        plt.ylabel("Score")
        plt.title("DeepShiva Performance on MBPP")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"mbpp_pass_at_k.{self.file_format}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        
        # Create comparison chart if requested
        if self.include_comparison and "pass@1" in pass_at_k:
            self._create_comparison_chart(
                benchmark="mbpp",
                metric="pass@1",
                deepshiva_value=pass_at_k["pass@1"],
                title="MBPP pass@1 Comparison",
                ylabel="pass@1 Score",
            )
    
    def visualize_gsm8k(self) -> None:
        """Visualize GSM8K results."""
        if "gsm8k" not in self.results:
            return
        
        results = self.results["gsm8k"]
        accuracy = results["accuracy"]
        
        # Create bar chart for accuracy
        plt.figure(figsize=(8, 6))
        
        # Plot
        bar = plt.bar(
            ["DeepShiva"],
            [accuracy],
            color=self.COLORS["DeepShiva"],
            alpha=0.8,
            width=0.5,
        )
        
        # Add value label
        plt.text(
            0,
            accuracy + 0.01,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        
        plt.ylim(0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("DeepShiva Performance on GSM8K")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"gsm8k_accuracy.{self.file_format}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        
        # Create comparison chart if requested
        if self.include_comparison:
            self._create_comparison_chart(
                benchmark="gsm8k",
                metric="accuracy",
                deepshiva_value=accuracy,
                title="GSM8K Accuracy Comparison",
                ylabel="Accuracy",
            )
    
    def visualize_model_comparison(self) -> None:
        """Create a radar chart comparing DeepShiva to other models across benchmarks."""
        # Collect metrics for comparison
        metrics = []
        deepshiva_values = []
        
        if "humaneval" in self.results and "pass@1" in self.results["humaneval"]["pass_at_k"]:
            metrics.append("HumanEval\npass@1")
            deepshiva_values.append(self.results["humaneval"]["pass_at_k"]["pass@1"])
        
        if "mbpp" in self.results and "pass@1" in self.results["mbpp"]["pass_at_k"]:
            metrics.append("MBPP\npass@1")
            deepshiva_values.append(self.results["mbpp"]["pass_at_k"]["pass@1"])
        
        if "gsm8k" in self.results:
            metrics.append("GSM8K\naccuracy")
            deepshiva_values.append(self.results["gsm8k"]["accuracy"])
        
        if len(metrics) < 2:
            return  # Need at least 2 metrics for a meaningful comparison
        
        # Select baseline models to include
        baseline_models = ["DeepSeek-Coder-V2-33B", "GPT-4", "CodeLlama-34B"]
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add DeepShiva values
        values = deepshiva_values + [deepshiva_values[0]]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label="DeepShiva", color=self.COLORS["DeepShiva"])
        ax.fill(angles, values, alpha=0.1, color=self.COLORS["DeepShiva"])
        
        # Add baseline models
        for model in baseline_models:
            model_values = []
            for i, metric in enumerate(metrics):
                if metric == "HumanEval\npass@1" and "humaneval" in self.BASELINE_MODELS and "pass@1" in self.BASELINE_MODELS["humaneval"]:
                    model_values.append(self.BASELINE_MODELS["humaneval"]["pass@1"].get(model, 0))
                elif metric == "MBPP\npass@1" and "mbpp" in self.BASELINE_MODELS and "pass@1" in self.BASELINE_MODELS["mbpp"]:
                    model_values.append(self.BASELINE_MODELS["mbpp"]["pass@1"].get(model, 0))
                elif metric == "GSM8K\naccuracy" and "gsm8k" in self.BASELINE_MODELS and "accuracy" in self.BASELINE_MODELS["gsm8k"]:
                    model_values.append(self.BASELINE_MODELS["gsm8k"]["accuracy"].get(model, 0))
                else:
                    model_values.append(0)
            
            # Close the loop
            model_values += [model_values[0]]
            
            # Plot
            ax.plot(angles, model_values, 'o-', linewidth=2, label=model, color=self.COLORS[model])
            ax.fill(angles, model_values, alpha=0.1, color=self.COLORS[model])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Set y-limits
        ax.set_ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("DeepShiva vs. Other Models", size=15, y=1.1)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"model_comparison_radar.{self.file_format}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        
        # Create bar chart comparison for all metrics
        self._create_overall_comparison_chart(metrics, deepshiva_values, baseline_models)
    
    def _create_comparison_chart(
        self,
        benchmark: str,
        metric: str,
        deepshiva_value: float,
        title: str,
        ylabel: str,
    ) -> None:
        """Create a comparison chart for a specific benchmark and metric.
        
        Args:
            benchmark: Benchmark name
            metric: Metric name
            deepshiva_value: DeepShiva's value for this metric
            title: Chart title
            ylabel: Y-axis label
        """
        if (
            benchmark not in self.BASELINE_MODELS or
            metric not in self.BASELINE_MODELS[benchmark]
        ):
            return
        
        # Get baseline values
        baseline_values = self.BASELINE_MODELS[benchmark][metric]
        
        # Create DataFrame for plotting
        models = ["DeepShiva"] + list(baseline_values.keys())
        values = [deepshiva_value] + list(baseline_values.values())
        
        # Sort by performance
        sorted_data = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
        models = [m for m, _ in sorted_data]
        values = [v for _, v in sorted_data]
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        
        # Get colors
        colors = [self.COLORS.get(model, "#999999") for model in models]
        
        # Plot
        bars = plt.bar(models, values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        plt.ylim(0, 1.0)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=30, ha="right")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"{benchmark}_{metric}_comparison.{self.file_format}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    
    def _create_overall_comparison_chart(
        self,
        metrics: List[str],
        deepshiva_values: List[float],
        baseline_models: List[str],
    ) -> None:
        """Create an overall comparison bar chart for all metrics.
        
        Args:
            metrics: List of metrics
            deepshiva_values: DeepShiva's values for these metrics
            baseline_models: List of baseline models to include
        """
        # Create DataFrame for plotting
        data = []
        
        # Add DeepShiva data
        for i, metric in enumerate(metrics):
            data.append({
                "Model": "DeepShiva",
                "Metric": metric,
                "Score": deepshiva_values[i],
            })
        
        # Add baseline models data
        for model in baseline_models:
            for i, metric in enumerate(metrics):
                score = 0
                if metric == "HumanEval\npass@1" and "humaneval" in self.BASELINE_MODELS and "pass@1" in self.BASELINE_MODELS["humaneval"]:
                    score = self.BASELINE_MODELS["humaneval"]["pass@1"].get(model, 0)
                elif metric == "MBPP\npass@1" and "mbpp" in self.BASELINE_MODELS and "pass@1" in self.BASELINE_MODELS["mbpp"]:
                    score = self.BASELINE_MODELS["mbpp"]["pass@1"].get(model, 0)
                elif metric == "GSM8K\naccuracy" and "gsm8k" in self.BASELINE_MODELS and "accuracy" in self.BASELINE_MODELS["gsm8k"]:
                    score = self.BASELINE_MODELS["gsm8k"]["accuracy"].get(model, 0)
                
                data.append({
                    "Model": model,
                    "Metric": metric,
                    "Score": score,
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        plt.figure(figsize=(14, 8))
        
        # Set up the plot
        ax = plt.subplot(111)
        
        # Width of a bar 
        bar_width = 0.15
        
        # Positions of the bars on the x-axis
        models = ["DeepShiva"] + baseline_models
        positions = np.arange(len(metrics))
        
        # Plot bars
        for i, model in enumerate(models):
            model_data = df[df["Model"] == model]
            model_scores = [model_data[model_data["Metric"] == metric]["Score"].values[0] if len(model_data[model_data["Metric"] == metric]) > 0 else 0 for metric in metrics]
            
            bars = plt.bar(
                positions + i * bar_width,
                model_scores,
                bar_width,
                alpha=0.8,
                label=model,
                color=self.COLORS.get(model, "#999999"),
            )
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
        
        # Set x-axis labels
        plt.xlabel("Benchmark Metrics")
        plt.ylabel("Score")
        plt.title("DeepShiva vs. Other Models Across Benchmarks")
        plt.xticks(positions + bar_width * (len(models) - 1) / 2, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"overall_comparison.{self.file_format}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    args = parse_args()
    
    # Create visualizer
    visualizer = ResultsVisualizer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        file_format=args.format,
        include_comparison=args.comparison,
    )
    
    # Generate visualizations
    visualizer.visualize_all()
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
