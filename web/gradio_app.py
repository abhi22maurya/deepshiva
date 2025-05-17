"""
DeepShiva Gradio Web Interface

This module provides a user-friendly web interface for the DeepShiva model using Gradio.
It supports:
- Text generation
- Code completion with syntax highlighting
- Mathematical reasoning
- Translation between languages
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Any

import gradio as gr
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import DeepShivaInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShiva Gradio Web Interface")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("DEEPSHIVA_MODEL_PATH", "models/pretrained/deepshiva-moe"),
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEEPSHIVA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run inference on",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=os.environ.get("DEEPSHIVA_PRECISION", "fp16"),
        choices=["fp32", "fp16", "int8", "int4"],
        help="Precision to use for inference",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DEEPSHIVA_UI_PORT", 7860)),
        help="Port to run the Gradio app on",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="soft",
        help="Gradio theme to use",
    )
    return parser.parse_args()


def create_gradio_app(inference_engine: DeepShivaInference):
    """Create the Gradio app for DeepShiva."""
    
    # Text Generation Tab
    with gr.Blocks() as text_generation_tab:
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                )
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    clear_button = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            max_new_tokens = gr.Slider(
                                minimum=1,
                                maximum=4096,
                                value=512,
                                step=1,
                                label="Max New Tokens",
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature",
                            )
                        with gr.Column():
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p",
                            )
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Top-k",
                            )
                            repetition_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.1,
                                step=0.05,
                                label="Repetition Penalty",
                            )
            
            with gr.Column(scale=3):
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=20,
                    interactive=False,
                )
        
        # Define generation function
        def generate_text(
            prompt: str,
            max_tokens: int,
            temp: float,
            p: float,
            k: int,
            rep_penalty: float,
        ) -> str:
            try:
                return inference_engine.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=p,
                    top_k=k,
                    repetition_penalty=rep_penalty,
                )
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                return f"Error: {str(e)}"
        
        # Connect UI components
        generate_button.click(
            generate_text,
            inputs=[
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            ],
            outputs=output_text,
        )
        
        clear_button.click(
            lambda: ["", 512, 0.7, 0.9, 50, 1.1, ""],
            inputs=None,
            outputs=[
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                output_text,
            ],
        )
    
    # Code Completion Tab
    with gr.Blocks() as code_completion_tab:
        with gr.Row():
            with gr.Column(scale=3):
                language_dropdown = gr.Dropdown(
                    choices=[
                        "python", "javascript", "typescript", "java", "c", "cpp", 
                        "csharp", "go", "rust", "php", "ruby", "swift", "kotlin",
                        "scala", "html", "css", "sql"
                    ],
                    value="python",
                    label="Programming Language",
                )
                code_input = gr.Code(
                    label="Code",
                    language="python",
                    lines=10,
                    value="def fibonacci(n):\n    # Complete this function to return the nth Fibonacci number\n    ",
                )
                with gr.Row():
                    complete_button = gr.Button("Complete Code", variant="primary")
                    code_clear_button = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            code_max_tokens = gr.Slider(
                                minimum=1,
                                maximum=2048,
                                value=512,
                                step=1,
                                label="Max New Tokens",
                            )
                            code_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.2,
                                step=0.05,
                                label="Temperature",
                            )
                        with gr.Column():
                            code_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                label="Top-p",
                            )
                            code_repetition_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.05,
                                step=0.05,
                                label="Repetition Penalty",
                            )
            
            with gr.Column(scale=3):
                code_output = gr.Code(
                    label="Completed Code",
                    language="python",
                    lines=20,
                    interactive=False,
                )
        
        # Update code language when dropdown changes
        def update_language(lang):
            return gr.Code.update(language=lang)
        
        language_dropdown.change(
            update_language,
            inputs=language_dropdown,
            outputs=code_input,
        )
        
        language_dropdown.change(
            update_language,
            inputs=language_dropdown,
            outputs=code_output,
        )
        
        # Define code completion function
        def complete_code_fn(
            code: str,
            language: str,
            max_tokens: int,
            temp: float,
            p: float,
            rep_penalty: float,
        ) -> str:
            try:
                completed = inference_engine.complete_code(
                    code_prompt=code,
                    language=language,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=p,
                    repetition_penalty=rep_penalty,
                )
                # Return the full code (prompt + completion)
                return code + completed
            except Exception as e:
                logger.error(f"Error completing code: {e}")
                return f"Error: {str(e)}"
        
        # Connect UI components
        complete_button.click(
            complete_code_fn,
            inputs=[
                code_input,
                language_dropdown,
                code_max_tokens,
                code_temperature,
                code_top_p,
                code_repetition_penalty,
            ],
            outputs=code_output,
        )
        
        code_clear_button.click(
            lambda: ["python", "def fibonacci(n):\n    # Complete this function to return the nth Fibonacci number\n    ", 512, 0.2, 0.95, 1.05, ""],
            inputs=None,
            outputs=[
                language_dropdown,
                code_input,
                code_max_tokens,
                code_temperature,
                code_top_p,
                code_repetition_penalty,
                code_output,
            ],
        )
    
    # Math Solving Tab
    with gr.Blocks() as math_tab:
        with gr.Row():
            with gr.Column(scale=3):
                math_input = gr.Textbox(
                    label="Math Problem",
                    placeholder="Enter a math problem here...",
                    lines=5,
                    value="If a train travels at 120 km/h and another train travels at 180 km/h in the opposite direction, how long will it take for them to be 450 km apart if they start at the same point?",
                )
                with gr.Row():
                    solve_button = gr.Button("Solve", variant="primary")
                    math_clear_button = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            math_max_tokens = gr.Slider(
                                minimum=1,
                                maximum=4096,
                                value=1024,
                                step=1,
                                label="Max New Tokens",
                            )
                            math_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                step=0.05,
                                label="Temperature",
                            )
                        with gr.Column():
                            show_work = gr.Checkbox(
                                label="Show Work",
                                value=True,
                            )
            
            with gr.Column(scale=3):
                math_output = gr.Textbox(
                    label="Solution",
                    lines=20,
                    interactive=False,
                )
        
        # Define math solving function
        def solve_math_fn(
            problem: str,
            max_tokens: int,
            temp: float,
            show_work_bool: bool,
        ) -> str:
            try:
                return inference_engine.solve_math(
                    problem=problem,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    show_work=show_work_bool,
                )
            except Exception as e:
                logger.error(f"Error solving math problem: {e}")
                return f"Error: {str(e)}"
        
        # Connect UI components
        solve_button.click(
            solve_math_fn,
            inputs=[
                math_input,
                math_max_tokens,
                math_temperature,
                show_work,
            ],
            outputs=math_output,
        )
        
        math_clear_button.click(
            lambda: ["If a train travels at 120 km/h and another train travels at 180 km/h in the opposite direction, how long will it take for them to be 450 km apart if they start at the same point?", 1024, 0.3, True, ""],
            inputs=None,
            outputs=[
                math_input,
                math_max_tokens,
                math_temperature,
                show_work,
                math_output,
            ],
        )
    
    # Translation Tab
    with gr.Blocks() as translation_tab:
        with gr.Row():
            with gr.Column(scale=3):
                source_lang = gr.Dropdown(
                    choices=["en", "hi", "ta", "bn", "te", "mr", "fr", "de", "es", "zh", "ja", "ko", "ru", "ar"],
                    value="en",
                    label="Source Language",
                )
                target_lang = gr.Dropdown(
                    choices=["en", "hi", "ta", "bn", "te", "mr", "fr", "de", "es", "zh", "ja", "ko", "ru", "ar"],
                    value="hi",
                    label="Target Language",
                )
                source_text = gr.Textbox(
                    label="Source Text",
                    placeholder="Enter text to translate...",
                    lines=5,
                    value="The DeepShiva model is designed to excel at code generation, mathematical reasoning, and multilingual support.",
                )
                with gr.Row():
                    translate_button = gr.Button("Translate", variant="primary")
                    translation_clear_button = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            translation_max_tokens = gr.Slider(
                                minimum=1,
                                maximum=2048,
                                value=512,
                                step=1,
                                label="Max New Tokens",
                            )
                            translation_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Temperature",
                            )
            
            with gr.Column(scale=3):
                translated_text = gr.Textbox(
                    label="Translated Text",
                    lines=20,
                    interactive=False,
                )
        
        # Define translation function
        def translate_fn(
            text: str,
            src_lang: str,
            tgt_lang: str,
            max_tokens: int,
            temp: float,
        ) -> str:
            try:
                return inference_engine.translate(
                    text=text,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                )
            except Exception as e:
                logger.error(f"Error translating text: {e}")
                return f"Error: {str(e)}"
        
        # Connect UI components
        translate_button.click(
            translate_fn,
            inputs=[
                source_text,
                source_lang,
                target_lang,
                translation_max_tokens,
                translation_temperature,
            ],
            outputs=translated_text,
        )
        
        translation_clear_button.click(
            lambda: ["en", "hi", "The DeepShiva model is designed to excel at code generation, mathematical reasoning, and multilingual support.", 512, 0.5, ""],
            inputs=None,
            outputs=[
                source_lang,
                target_lang,
                source_text,
                translation_max_tokens,
                translation_temperature,
                translated_text,
            ],
        )
    
    # Create the main app with tabs
    demo = gr.TabbedInterface(
        [text_generation_tab, code_completion_tab, math_tab, translation_tab],
        ["Text Generation", "Code Completion", "Math Solving", "Translation"],
        title="DeepShiva: Mixture-of-Experts LLM",
        description="DeepShiva is an open-source, Mixture-of-Experts (MoE) code language model designed to excel in coding tasks, mathematical reasoning, and multilingual support, particularly for Indian languages.",
    )
    
    return demo


def main():
    args = parse_args()
    
    # Load the model
    try:
        logger.info(f"Loading model from {args.model_path}")
        inference_engine = DeepShivaInference(
            model_path=args.model_path,
            device=args.device,
            precision=args.precision,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        inference_engine = None
        raise e
    
    # Create and launch the Gradio app
    app = create_gradio_app(inference_engine)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=args.theme,
    )


if __name__ == "__main__":
    main()
