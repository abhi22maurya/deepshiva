# DeepShiva Evaluation Framework

This directory contains a comprehensive evaluation framework for benchmarking the DeepShiva model across multiple dimensions:

- **Code Generation**: HumanEval and MBPP benchmarks
- **Mathematical Reasoning**: GSM8K benchmark
- **Multilingual Capabilities**: Translation, multilingual code, and question answering

## Overview

The evaluation framework is designed to provide a thorough assessment of DeepShiva's capabilities compared to other state-of-the-art models like DeepSeek-Coder-V2, CodeLlama, GPT-4, and others.

## Scripts

- `evaluate.py`: Evaluates DeepShiva on code and math benchmarks (HumanEval, MBPP, GSM8K)
- `evaluate_multilingual.py`: Evaluates DeepShiva on multilingual tasks (translation, code, QA)
- `visualize_results.py`: Generates visualizations of benchmark results
- `generate_report.py`: Creates a comprehensive markdown report of all evaluation results
- `run_all_benchmarks.py`: Orchestrates running all benchmarks in sequence

## Usage

### Running All Benchmarks

To run all benchmarks with default settings:

```bash
python evaluation/run_all_benchmarks.py --model_path /path/to/model --device cuda --generate_report --visualize
```

### Running Individual Benchmarks

#### Code Benchmarks

```bash
python evaluation/evaluate.py --model_path /path/to/model --benchmark humaneval --device cuda
python evaluation/evaluate.py --model_path /path/to/model --benchmark mbpp --device cuda
```

#### Math Benchmarks

```bash
python evaluation/evaluate.py --model_path /path/to/model --benchmark gsm8k --device cuda
```

#### Multilingual Benchmarks

```bash
python evaluation/evaluate_multilingual.py --model_path /path/to/model --task translation --languages en,hi,bn,ta,te
python evaluation/evaluate_multilingual.py --model_path /path/to/model --task code --languages en,hi,bn,ta,te
python evaluation/evaluate_multilingual.py --model_path /path/to/model --task qa --languages en,hi,bn,ta,te
```

### Generating Visualizations

```bash
python evaluation/visualize_results.py --results_dir evaluation/results --output_dir evaluation/visualizations
```

### Generating Reports

```bash
python evaluation/generate_report.py --results_dir evaluation/results --visualizations_dir evaluation/visualizations --output_file evaluation/reports/DeepShiva_Evaluation_Report.md
```

## Metrics

The framework evaluates DeepShiva using the following metrics:

### Code Generation
- **pass@k**: Probability of solving a problem with k samples (HumanEval, MBPP)

### Mathematical Reasoning
- **Accuracy**: Percentage of correctly solved problems (GSM8K)

### Multilingual Capabilities
- **BLEU**: Translation quality metric
- **Similarity**: Code similarity metric for multilingual code generation
- **F1 & Exact Match**: Metrics for multilingual question answering

## Directory Structure

```
evaluation/
├── README.md                   # This file
├── evaluate.py                 # Code and math evaluation script
├── evaluate_multilingual.py    # Multilingual evaluation script
├── visualize_results.py        # Visualization script
├── generate_report.py          # Report generation script
├── run_all_benchmarks.py       # Orchestration script
├── results/                    # Evaluation results (JSON)
├── visualizations/             # Generated charts and graphs
└── reports/                    # Comprehensive evaluation reports
```

## Requirements

The evaluation framework requires the following dependencies:

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- SacreBLEU (for translation evaluation)
- Matplotlib and Seaborn (for visualizations)
- Pandas
- NumPy
- Tabulate (for report generation)

All dependencies are included in the main `requirements.txt` file at the project root.

## Extending the Framework

To add new benchmarks:

1. Create a new evaluator class that inherits from the base `Evaluator` class
2. Implement the `evaluate()` method to run the benchmark
3. Add the new benchmark to the appropriate evaluation script
4. Update the `run_all_benchmarks.py` script to include the new benchmark

## Citation

If you use this evaluation framework in your research, please cite:

```
@misc{deepshiva2023,
  author = {DeepShiva Team},
  title = {DeepShiva: Open-Source Mixture-of-Experts Model for Code and Multilingual Tasks},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/username/deepshiva}
}
```
