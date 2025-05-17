"""
DeepShiva Web Interface

A user-friendly web interface for interacting with the DeepShiva model.
"""

import gradio as gr
from optimized_llm import OptimizedLLM
import torch
import gc
import os
from typing import Dict, Any, Optional

# Configuration
DEFAULT_MODEL = "codellama/CodeLlama-7b-hf"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1

# Global model instance
llm_instance = None

def load_model(
    model_name: str,
    device: str = "auto",
    load_in_8bit: bool = True,
    progress: gr.Progress = None
) -> Dict[str, Any]:
    """Load the model with the given configuration."""
    global llm_instance
    
    try:
        # Clear existing model if any
        if llm_instance is not None:
            del llm_instance
            torch.cuda.empty_cache()
            gc.collect()
        
        # Configure memory settings
        max_memory = {
            "mps:0": "4GB",
            "cpu": "8GB"
        }
        
        if progress is not None:
            progress(0.2, desc="Initializing model...")
        
        # Initialize the model
        llm_instance = OptimizedLLM(
            model_name=model_name,
            device=device,
            max_memory=max_memory,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16
        )
        
        if progress is not None:
            progress(0.8, desc="Warming up model...")
        
        # Test the model
        test_prompt = "Hello, "
        _ = llm_instance.generate(
            prompt=test_prompt,
            max_length=10,
            temperature=0.1,
            stream=False
        )
        
        if progress is not None:
            progress(1.0, desc="Model loaded successfully!")
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully!"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load model: {str(e)}"
        }

def generate_response(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    progress: gr.Progress = None
) -> str:
    """Generate a response from the model."""
    global llm_instance
    
    if llm_instance is None:
        return "Error: Model not loaded. Please load a model first."
    
    try:
        if progress is not None:
            progress(0.1, desc="Generating response...")
        
        # Generate response
        response = llm_instance.generate(
            prompt=prompt,
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stream=False
        )
        
        if progress is not None:
            progress(1.0, desc="Generation complete!")
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    # Use the most basic Blocks with no custom CSS
    with gr.Blocks(title="DeepShiva - AI Code Assistant") as app:
        # Header
        gr.Markdown(
            """
            # 🚀 DeepShiva AI Code Assistant
            An intelligent coding assistant powered by CodeLlama.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Model settings
                with gr.Accordion("⚙️ Model Settings", open=False):
                    model_name = gr.Textbox(
                        label="Model Name/Path",
                        value=DEFAULT_MODEL,
                        placeholder="Enter model name or path...",
                    )
                    
                    with gr.Row():
                        load_btn = gr.Button("Load Model", variant="primary")
                        unload_btn = gr.Button("Unload Model", variant="secondary")
                    
                    model_status = gr.Markdown("*No model loaded*")
                
                # Generation settings
                with gr.Accordion("🎛️ Generation Settings", open=True):
                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=2048,
                        value=DEFAULT_MAX_TOKENS,
                        step=32,
                        label="Max Tokens"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=DEFAULT_TEMPERATURE,
                        step=0.1,
                        label="Temperature"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=DEFAULT_TOP_P,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)"
                    )
                    
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=DEFAULT_TOP_K,
                        step=1,
                        label="Top-k Sampling"
                    )
                    
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=DEFAULT_REPETITION_PENALTY,
                        step=0.1,
                        label="Repetition Penalty"
                    )
                
                # Prompt input
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=8,
                    max_lines=20,
                )
                
                # Generate button
                generate_btn = gr.Button("Generate", variant="primary")
            
            # Output
            with gr.Column(scale=7):
                output = gr.Textbox(
                    label="Response",
                    interactive=False,
                    lines=25,
                    show_copy_button=True,
                    container=True,
                )
        
        # Examples
        examples = [
            ["def fibonacci(n):", 100, 0.2, 0.9, 50, 1.1],
            ["Explain how to implement a binary search tree in Python.", 300, 0.7, 0.9, 50, 1.1],
            ["Write a Python function to sort a list of dictionaries by a specific key.", 200, 0.5, 0.9, 50, 1.1],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[prompt, max_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=output,
            fn=generate_response,
            cache_examples=False,
            label="Examples (click to load)",
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### About DeepShiva
            DeepShiva is an open-source AI coding assistant built with ❤️.
            """
        )
        
        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=[model_name],
            outputs=model_status,
            show_progress="full",
            api_name="load_model"
        )
        
        unload_btn.click(
            fn=lambda: {"status": "success", "message": "Model unloaded successfully!"},
            outputs=model_status,
            show_progress="hidden"
        )
        
        generate_btn.click(
            fn=generate_response,
            inputs=[
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty
            ],
            outputs=output,
            show_progress="full"
        )
        
        # Handle Enter key in prompt
        prompt.submit(
            fn=generate_response,
            inputs=[
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty
            ],
            outputs=output,
            show_progress="full"
        )
    
    return app

if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    
    # Simple launch with minimal settings
    try:
        print("Launching Gradio interface...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=True,
            show_api=False
        )
    except Exception as e:
        print(f"Error launching Gradio: {str(e)}")
        print("Trying with more basic settings...")
        app.launch(server_name="0.0.0.0")
