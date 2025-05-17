"""
Enhanced DeepShiva Gradio Web Interface

This module provides an improved web interface for the DeepShiva model with:
- Streaming responses for faster user experience
- Advanced visualization options
- Multiple task interfaces (text generation, code completion, math solving, translation)
- Performance metrics display
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Iterator, Tuple

import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import DeepShiva inference module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.enhanced_inference import EnhancedInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
inference_engine = None
generation_times = []
token_counts = []


def load_model():
    """Load the DeepShiva model."""
    global inference_engine
    
    # Get model path from environment variable or use default
    model_path = os.environ.get("DEEPSHIVA_MODEL_PATH", "models/pretrained/deepshiva-moe")
    device = os.environ.get("DEEPSHIVA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    precision = os.environ.get("DEEPSHIVA_PRECISION", "fp16")
    use_bettertransformer = os.environ.get("DEEPSHIVA_USE_BETTERTRANSFORMER", "true").lower() == "true"
    use_compile = os.environ.get("DEEPSHIVA_USE_COMPILE", "false").lower() == "true"
    
    try:
        logger.info(f"Loading model from {model_path}")
        inference_engine = EnhancedInference(
            model_path=model_path,
            device=device,
            precision=precision,
            use_bettertransformer=use_bettertransformer,
            use_compile=use_compile,
        )
        logger.info("Model loaded successfully")
        return f"Model loaded successfully from {model_path} on {device} with {precision} precision"
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        logger.error(error_msg)
        return error_msg


def generate_text(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    stream: bool = True,
) -> Iterator[str]:
    """Generate text based on the prompt with streaming support."""
    global inference_engine, generation_times, token_counts
    
    if inference_engine is None:
        yield "Model not loaded. Please load the model first."
        return
    
    try:
        # Record start time
        start_time = time.time()
        
        # Generate text
        generated_text = ""
        for token in inference_engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stream=stream,
        ):
            generated_text += token
            yield generated_text
        
        # Record generation time and token count
        generation_time = time.time() - start_time
        token_count = len(inference_engine.tokenizer.encode(generated_text))
        
        generation_times.append(generation_time)
        token_counts.append(token_count)
        
        logger.info(f"Generated {token_count} tokens in {generation_time:.2f} seconds")
    except Exception as e:
        error_msg = f"Error generating text: {e}"
        logger.error(error_msg)
        yield error_msg


def complete_code(
    code_prompt: str,
    language: str = "python",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    stream: bool = True,
) -> Iterator[str]:
    """Complete code based on the prompt with streaming support."""
    global inference_engine, generation_times, token_counts
    
    if inference_engine is None:
        yield "Model not loaded. Please load the model first."
        return
    
    try:
        # Record start time
        start_time = time.time()
        
        # Generate code
        generated_code = ""
        for token in inference_engine.complete_code(
            code_prompt=code_prompt,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
        ):
            generated_code += token
            yield generated_code
        
        # Record generation time and token count
        generation_time = time.time() - start_time
        token_count = len(inference_engine.tokenizer.encode(generated_code))
        
        generation_times.append(generation_time)
        token_counts.append(token_count)
        
        logger.info(f"Generated {token_count} tokens in {generation_time:.2f} seconds")
    except Exception as e:
        error_msg = f"Error completing code: {e}"
        logger.error(error_msg)
        yield error_msg


def solve_math(
    problem: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    show_work: bool = True,
    stream: bool = True,
) -> Iterator[str]:
    """Solve a mathematical problem with streaming support."""
    global inference_engine, generation_times, token_counts
    
    if inference_engine is None:
        yield "Model not loaded. Please load the model first."
        return
    
    try:
        # Record start time
        start_time = time.time()
        
        # Generate solution
        solution = ""
        for token in inference_engine.solve_math(
            problem=problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_work=show_work,
            stream=stream,
        ):
            solution += token
            yield solution
        
        # Record generation time and token count
        generation_time = time.time() - start_time
        token_count = len(inference_engine.tokenizer.encode(solution))
        
        generation_times.append(generation_time)
        token_counts.append(token_count)
        
        logger.info(f"Generated {token_count} tokens in {generation_time:.2f} seconds")
    except Exception as e:
        error_msg = f"Error solving math problem: {e}"
        logger.error(error_msg)
        yield error_msg


def translate(
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    stream: bool = True,
) -> Iterator[str]:
    """Translate text from source language to target language with streaming support."""
    global inference_engine, generation_times, token_counts
    
    if inference_engine is None:
        yield "Model not loaded. Please load the model first."
        return
    
    try:
        # Record start time
        start_time = time.time()
        
        # Generate translation
        translation = ""
        for token in inference_engine.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
        ):
            translation += token
            yield translation
        
        # Record generation time and token count
        generation_time = time.time() - start_time
        token_count = len(inference_engine.tokenizer.encode(translation))
        
        generation_times.append(generation_time)
        token_counts.append(token_count)
        
        logger.info(f"Generated {token_count} tokens in {generation_time:.2f} seconds")
    except Exception as e:
        error_msg = f"Error translating text: {e}"
        logger.error(error_msg)
        yield error_msg


def plot_performance_metrics() -> Figure:
    """Plot performance metrics."""
    global generation_times, token_counts
    
    if not generation_times or not token_counts:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Calculate tokens per second
    tokens_per_second = [count / time for count, time in zip(token_counts, generation_times)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot generation times
    ax1.plot(generation_times, marker="o", linestyle="-", color="blue")
    ax1.set_title("Generation Times")
    ax1.set_xlabel("Request Index")
    ax1.set_ylabel("Time (seconds)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # Plot tokens per second
    ax2.plot(tokens_per_second, marker="o", linestyle="-", color="green")
    ax2.set_title("Tokens per Second")
    ax2.set_xlabel("Request Index")
    ax2.set_ylabel("Tokens/s")
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    # Add average lines
    avg_time = np.mean(generation_times)
    avg_tps = np.mean(tokens_per_second)
    
    ax1.axhline(y=avg_time, color="red", linestyle="--", alpha=0.7, label=f"Avg: {avg_time:.2f}s")
    ax2.axhline(y=avg_tps, color="red", linestyle="--", alpha=0.7, label=f"Avg: {avg_tps:.2f} tokens/s")
    
    ax1.legend()
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_gradio_app():
    """Create and configure the Gradio app."""
    # CSS for styling
    css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
    }
    """
    
    # Create Gradio blocks
    with gr.Blocks(css=css) as app:
        gr.Markdown("# DeepShiva MoE Model", elem_classes=["title"])
        gr.Markdown("### An advanced Mixture-of-Experts language model for text generation, code completion, math solving, and translation", elem_classes=["subtitle"])
        
        # Model loading section
        with gr.Row():
            with gr.Column():
                load_button = gr.Button("Load Model")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                load_button.click(load_model, inputs=[], outputs=model_status)
        
        # Main interface with tabs
        with gr.Tabs():
            # Text Generation tab
            with gr.TabItem("Text Generation"):
                with gr.Row():
                    with gr.Column():
                        text_prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter your prompt here...")
                        with gr.Row():
                            text_max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens")
                            text_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                        with gr.Row():
                            text_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")
                            text_top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
                        with gr.Row():
                            text_repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
                            text_do_sample = gr.Checkbox(value=True, label="Use Sampling")
                        text_generate_button = gr.Button("Generate")
                    with gr.Column():
                        text_output = gr.Textbox(label="Generated Text", lines=20, interactive=False)
                
                text_generate_button.click(
                    generate_text,
                    inputs=[
                        text_prompt,
                        text_max_tokens,
                        text_temperature,
                        text_top_p,
                        text_top_k,
                        text_repetition_penalty,
                        text_do_sample,
                    ],
                    outputs=text_output,
                )
            
            # Code Completion tab
            with gr.TabItem("Code Completion"):
                with gr.Row():
                    with gr.Column():
                        code_prompt = gr.Code(label="Code Prompt", language="python", lines=10)
                        with gr.Row():
                            code_language = gr.Dropdown(
                                choices=["python", "javascript", "java", "c", "cpp", "rust", "go", "ruby", "php", "typescript", "html", "css", "sql"],
                                value="python",
                                label="Language",
                            )
                            code_max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens")
                        with gr.Row():
                            code_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.05, label="Temperature")
                            code_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")
                        code_repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.05, step=0.05, label="Repetition Penalty")
                        code_generate_button = gr.Button("Complete Code")
                    with gr.Column():
                        code_output = gr.Code(label="Completed Code", language="python", lines=20)
                
                # Update code output language when input language changes
                code_language.change(
                    lambda lang: gr.update(language=lang),
                    inputs=code_language,
                    outputs=code_output,
                )
                
                code_generate_button.click(
                    complete_code,
                    inputs=[
                        code_prompt,
                        code_language,
                        code_max_tokens,
                        code_temperature,
                        code_top_p,
                        code_repetition_penalty,
                    ],
                    outputs=code_output,
                )
            
            # Math Solving tab
            with gr.TabItem("Math Solving"):
                with gr.Row():
                    with gr.Column():
                        math_problem = gr.Textbox(label="Math Problem", lines=5, placeholder="Enter a mathematical problem...")
                        with gr.Row():
                            math_max_tokens = gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max New Tokens")
                            math_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Temperature")
                        math_show_work = gr.Checkbox(value=True, label="Show Work")
                        math_solve_button = gr.Button("Solve")
                    with gr.Column():
                        math_output = gr.Textbox(label="Solution", lines=20, interactive=False)
                
                math_solve_button.click(
                    solve_math,
                    inputs=[
                        math_problem,
                        math_max_tokens,
                        math_temperature,
                        math_show_work,
                    ],
                    outputs=math_output,
                )
            
            # Translation tab
            with gr.TabItem("Translation"):
                with gr.Row():
                    with gr.Column():
                        translation_text = gr.Textbox(label="Text to Translate", lines=5, placeholder="Enter text to translate...")
                        with gr.Row():
                            translation_source_lang = gr.Dropdown(
                                choices=["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Russian", "Arabic", "Portuguese"],
                                value="English",
                                label="Source Language",
                            )
                            translation_target_lang = gr.Dropdown(
                                choices=["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Russian", "Arabic", "Portuguese"],
                                value="Hindi",
                                label="Target Language",
                            )
                        with gr.Row():
                            translation_max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens")
                            translation_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Temperature")
                        translation_button = gr.Button("Translate")
                    with gr.Column():
                        translation_output = gr.Textbox(label="Translation", lines=20, interactive=False)
                
                translation_button.click(
                    translate,
                    inputs=[
                        translation_text,
                        translation_source_lang,
                        translation_target_lang,
                        translation_max_tokens,
                        translation_temperature,
                    ],
                    outputs=translation_output,
                )
            
            # Performance Metrics tab
            with gr.TabItem("Performance Metrics"):
                with gr.Row():
                    refresh_metrics_button = gr.Button("Refresh Metrics")
                    metrics_plot = gr.Plot(label="Performance Metrics")
                
                refresh_metrics_button.click(
                    plot_performance_metrics,
                    inputs=[],
                    outputs=metrics_plot,
                )
        
        # Footer
        gr.Markdown("#### DeepShiva MoE Model - Powered by Enhanced Inference Engine", elem_classes=["footer"])
    
    return app


if __name__ == "__main__":
    # Create and launch the app
    app = create_gradio_app()
    
    # Get Gradio server port from environment variable or use default
    port = int(os.environ.get("DEEPSHIVA_GRADIO_PORT", 7860))
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
