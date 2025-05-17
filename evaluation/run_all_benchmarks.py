#!/usr/bin/env python3
"""
DeepShiva Benchmark Runner

This script runs all DeepShiva evaluation benchmarks in sequence:
- Code benchmarks (HumanEval, MBPP)
- Math reasoning benchmarks (GSM8K)
- Multilingual benchmarks (Translation, Code, QA)

Results are saved to the specified output directory and a comprehensive report is generated.
"""

import os
import argparse
import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShiva Benchmark Runner")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (cuda or cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Precision to use for inference",
    )
    parser.add_argument(
        "--skip_code",
        action="store_true",
        help="Skip code benchmarks (HumanEval, MBPP)",
    )
    parser.add_argument(
        "--skip_math",
        action="store_true",
        help="Skip math benchmarks (GSM8K)",
    )
    parser.add_argument(
        "--skip_multilingual",
        action="store_true",
        help="Skip multilingual benchmarks",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel execution",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate a comprehensive report after evaluation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of results",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,hi,bn,ta,te",
        help="Comma-separated list of language codes for multilingual evaluation",
    )
    return parser.parse_args()


def run_command(cmd, cwd=None):
    """Run a command and log output.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        
    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        if return_code != 0:
            # Log error output
            stderr = process.stderr.read()
            logger.error(f"Command failed with return code {return_code}")
            logger.error(stderr)
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def run_code_benchmarks(args):
    """Run code benchmarks (HumanEval, MBPP).
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if benchmarks succeeded, False otherwise
    """
    logger.info("Running code benchmarks...")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run HumanEval benchmark
    humaneval_cmd = [
        "python", "evaluation/evaluate.py",
        "--model_path", args.model_path,
        "--benchmark", "humaneval",
        "--device", args.device,
        "--precision", args.precision,
        "--num_samples", "10",  # Use 10 samples for pass@k
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
    ]
    
    humaneval_success = run_command(humaneval_cmd)
    if not humaneval_success:
        logger.error("HumanEval benchmark failed")
    
    # Run MBPP benchmark
    mbpp_cmd = [
        "python", "evaluation/evaluate.py",
        "--model_path", args.model_path,
        "--benchmark", "mbpp",
        "--device", args.device,
        "--precision", args.precision,
        "--num_samples", "10",  # Use 10 samples for pass@k
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
    ]
    
    mbpp_success = run_command(mbpp_cmd)
    if not mbpp_success:
        logger.error("MBPP benchmark failed")
    
    return humaneval_success and mbpp_success


def run_math_benchmarks(args):
    """Run math benchmarks (GSM8K).
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if benchmarks succeeded, False otherwise
    """
    logger.info("Running math benchmarks...")
    
    # Run GSM8K benchmark
    gsm8k_cmd = [
        "python", "evaluation/evaluate.py",
        "--model_path", args.model_path,
        "--benchmark", "gsm8k",
        "--device", args.device,
        "--precision", args.precision,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
    ]
    
    gsm8k_success = run_command(gsm8k_cmd)
    if not gsm8k_success:
        logger.error("GSM8K benchmark failed")
    
    return gsm8k_success


def run_multilingual_benchmarks(args):
    """Run multilingual benchmarks.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if benchmarks succeeded, False otherwise
    """
    logger.info("Running multilingual benchmarks...")
    
    # Run translation benchmark
    translation_cmd = [
        "python", "evaluation/evaluate_multilingual.py",
        "--model_path", args.model_path,
        "--task", "translation",
        "--languages", args.languages,
        "--device", args.device,
        "--precision", args.precision,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
    ]
    
    translation_success = run_command(translation_cmd)
    if not translation_success:
        logger.error("Translation benchmark failed")
    
    # Run multilingual code benchmark
    code_cmd = [
        "python", "evaluation/evaluate_multilingual.py",
        "--model_path", args.model_path,
        "--task", "code",
        "--languages", args.languages,
        "--device", args.device,
        "--precision", args.precision,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
    ]
    
    code_success = run_command(code_cmd)
    if not code_success:
        logger.error("Multilingual code benchmark failed")
    
    # Run multilingual QA benchmark
    qa_cmd = [
        "python", "evaluation/evaluate_multilingual.py",
        "--model_path", args.model_path,
        "--task", "qa",
        "--languages", args.languages,
        "--device", args.device,
        "--precision", args.precision,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
    ]
    
    qa_success = run_command(qa_cmd)
    if not qa_success:
        logger.error("Multilingual QA benchmark failed")
    
    return translation_success and code_success and qa_success


def generate_visualizations(args):
    """Generate visualizations of benchmark results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if visualization succeeded, False otherwise
    """
    logger.info("Generating visualizations...")
    
    # Create visualizations directory
    visualizations_dir = os.path.join(args.output_dir, "../visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Run visualization script
    viz_cmd = [
        "python", "evaluation/visualize_results.py",
        "--results_dir", args.output_dir,
        "--output_dir", visualizations_dir,
        "--format", "png",
        "--comparison",
    ]
    
    viz_success = run_command(viz_cmd)
    if not viz_success:
        logger.error("Visualization generation failed")
    
    return viz_success


def generate_report(args):
    """Generate a comprehensive report of benchmark results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if report generation succeeded, False otherwise
    """
    logger.info("Generating evaluation report...")
    
    # Create report directory
    report_dir = os.path.join(args.output_dir, "../reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create timestamp for this report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"DeepShiva_Evaluation_Report_{timestamp}.md")
    
    # Create visualizations directory path
    visualizations_dir = os.path.join(args.output_dir, "../visualizations")
    
    # Run report generation script
    report_cmd = [
        "python", "evaluation/generate_report.py",
        "--results_dir", args.output_dir,
        "--visualizations_dir", visualizations_dir,
        "--output_file", report_file,
        "--model_path", args.model_path,
        "--include_comparison",
    ]
    
    report_success = run_command(report_cmd)
    if not report_success:
        logger.error("Report generation failed")
    else:
        logger.info(f"Report generated at {report_file}")
    
    return report_success


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Run benchmarks
    code_success = True
    math_success = True
    multilingual_success = True
    
    if not args.skip_code:
        code_success = run_code_benchmarks(args)
    
    if not args.skip_math:
        math_success = run_math_benchmarks(args)
    
    if not args.skip_multilingual:
        multilingual_success = run_multilingual_benchmarks(args)
    
    # Generate visualizations if requested
    if args.visualize:
        viz_success = generate_visualizations(args)
        if not viz_success:
            logger.warning("Visualization generation had issues")
    
    # Generate report if requested
    if args.generate_report:
        report_success = generate_report(args)
        if not report_success:
            logger.warning("Report generation had issues")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"All benchmarks completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Report overall status
    if code_success and math_success and multilingual_success:
        logger.info("All benchmarks completed successfully")
        return 0
    else:
        logger.warning("Some benchmarks had issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
