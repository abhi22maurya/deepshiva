"""
Minimal Flask-based interface for DeepShiva
"""

from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context
from optimized_llm import OptimizedLLM
import threading
import time
import logging
import os
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
llm_instance = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DeepShiva - AI Code Assistant</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1 { color: #2c3e50; }
        textarea, input[type="text"], button {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        button:disabled {
            background-color: #cccccc;
        }
        #output {
            border: 1px solid #ddd;
            padding: 15px;
            min-height: 200px;
            margin: 10px 0;
            white-space: pre-wrap;
            background-color: #f9f9f9;
            border-radius: 4px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>DeepShiva - AI Code Assistant</h1>
    
    <div>
        <label for="model_name">Model Name/Path:</label>
        <input type="text" id="model_name" value="stabilityai/stable-code-3b">
        <button onclick="loadModel()">Load Model</button>
        <div id="model_status" class="status">No model loaded</div>
    </div>
    
    <div>
        <label for="prompt">Prompt:</label>
        <textarea id="prompt" rows="6" placeholder="Enter your prompt here..."></textarea>
        <button onclick="generate()" id="generate_btn">Generate</button>
    </div>
    
    <div>
        <h3>Response:</h3>
        <div id="output">Response will appear here...</div>
    </div>
    
    <script>
        function loadModel() {
            const modelName = document.getElementById('model_name').value;
            const statusDiv = document.getElementById('model_status');
            const btn = document.querySelector('button[onclick="loadModel()"]');
            
            statusDiv.textContent = 'Loading model...';
            statusDiv.className = 'status';
            btn.disabled = true;
            
            fetch('/load_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_name: modelName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    statusDiv.textContent = data.message;
                    statusDiv.className = 'status success';
                } else {
                    statusDiv.textContent = 'Error: ' + data.message;
                    statusDiv.className = 'status error';
                }
                btn.disabled = false;
            })
            .catch(error => {
                statusDiv.textContent = 'Error: ' + error.message;
                statusDiv.className = 'status error';
                btn.disabled = false;
            });
        }
        
        function generate() {
            const prompt = document.getElementById('prompt').value;
            const outputDiv = document.getElementById('output');
            const btn = document.getElementById('generate_btn');
            
            if (!prompt) {
                outputDiv.textContent = 'Please enter a prompt';
                return;
            }
            
            outputDiv.textContent = 'Generating...';
            btn.disabled = true;
            
            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    prompt: prompt,
                    max_length: 512,
                    temperature: 0.7
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    outputDiv.textContent = data.response;
                } else {
                    outputDiv.textContent = 'Error: ' + data.message;
                }
                btn.disabled = false;
            })
            .catch(error => {
                outputDiv.textContent = 'Error: ' + error.message;
                btn.disabled = false;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/load_model', methods=['POST'])
def load_model():
    global llm_instance
    
    try:
        model_name = request.json.get('model_name', 'stabilityai/stable-code-3b')
        
        if llm_instance is not None:
            logger.info("Clearing previous model from memory")
            llm_instance.clear_memory()
            del llm_instance
            llm_instance = None
        
        # Determine best device
        device = "auto"
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
        elif torch.backends.mps.is_available():
            logger.info("MPS is available, using Apple Silicon")
        else:
            logger.info("Using CPU for inference")
        
        # Configure memory settings based on available hardware
        max_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                max_memory[f"cuda:{i}"] = "12GB"
        elif torch.backends.mps.is_available():
            max_memory["mps"] = "6GB"
            max_memory["cpu"] = "10GB"
        # If not CUDA or MPS, only CPU will be used and max_memory for cpu is sufficient
        
        logger.info(f"Loading model: {model_name}")
        llm_instance = OptimizedLLM(
            model_name=model_name,
            device=device,
            max_memory=max_memory,
            load_in_8bit=True,
            torch_dtype="auto"
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} loaded successfully!'
        })
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    global llm_instance
    
    if llm_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'No model loaded. Please load a model first.'
        }), 400
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = min(int(data.get('max_length', 512)), 512)  # Cap max length for better performance
        temperature = float(data.get('temperature', 0.7))
        
        if not prompt:
            return jsonify({
                'status': 'error',
                'message': 'Prompt cannot be empty'
            }), 400
        
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # Use a timeout to prevent hanging
        start_time = time.time()
        timeout = 60  # 60 second timeout
        
        # Generate response
        try:
            response = llm_instance.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            return jsonify({
                'status': 'success',
                'response': response,
                'generation_time_seconds': round(generation_time, 2)
            })
        except Exception as inner_e:
            logger.error(f"Error during generation: {str(inner_e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f"Generation error: {str(inner_e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/stream', methods=['POST'])
def stream_response():
    global llm_instance
    
    if llm_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'No model loaded. Please load a model first.'
        }), 400
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = min(int(data.get('max_length', 512)), 2048)
        temperature = float(data.get('temperature', 0.7))
        
        if not prompt:
            return jsonify({
                'status': 'error',
                'message': 'Prompt cannot be empty'
            }), 400
        
        def generate_stream():
            streamer = llm_instance.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                stream=True
            )
            
            for text in streamer:
                yield f"data: {text}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client
        
        return Response(
            stream_with_context(generate_stream()),
            mimetype='text/event-stream'
        )
    except Exception as e:
        logger.error(f"Error streaming response: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    global llm_instance
    
    if llm_instance is None:
        return jsonify({
            'loaded': False,
            'model_name': None
        })
    
    return jsonify({
        'loaded': True,
        'model_name': llm_instance.model_name,
        'device': str(llm_instance.device)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting DeepShiva minimal app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
