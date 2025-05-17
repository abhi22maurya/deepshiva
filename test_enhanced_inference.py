"""
Test script for the enhanced inference capabilities of DeepShiva.

This script demonstrates how to use the enhanced inference API without
requiring a pre-trained model by creating a mock model for testing.
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced inference module
from inference.enhanced_inference import EnhancedInference


class MockEnhancedDeepShivaMoE(nn.Module):
    """Mock model for testing the enhanced inference API."""
    
    def __init__(self):
        super().__init__()
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """Mock forward pass that returns logits and past key values."""
        batch_size, seq_len = input_ids.shape
        
        # Create mock logits
        if past_key_values is None:
            # First forward pass
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
        else:
            # Subsequent pass with only the last token
            logits = torch.randn(batch_size, 1, self.vocab_size)
        
        # Create mock past key values
        mock_past_kv = tuple(
            (
                torch.randn(batch_size, 32, seq_len, 128),
                torch.randn(batch_size, 32, seq_len, 128)
            )
            for _ in range(32)  # 32 layers
        )
        
        # Create a class with the expected output format
        class ModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values
            
            def __getitem__(self, key):
                if key == "logits":
                    return self.logits
                elif key == "past_key_values":
                    return self.past_key_values
                else:
                    raise KeyError(f"Key {key} not found")
        
        return ModelOutput(
            logits=logits,
            past_key_values=mock_past_kv if use_cache else None
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=100, **kwargs):
        """Mock generate method that returns a sequence of tokens."""
        batch_size = input_ids.shape[0]
        
        # Generate a sequence of tokens that's slightly longer than the input
        seq_len = input_ids.shape[1]
        new_len = min(seq_len + 20, max_length)
        
        # Create a sequence that starts with the input_ids and adds some random tokens
        generated = torch.cat([
            input_ids,
            torch.randint(100, 32000, (batch_size, new_len - seq_len))
        ], dim=1)
        
        return generated
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Mock from_pretrained method."""
        print(f"Mock loading model from {model_path}")
        return cls()


def create_mock_tokenizer(vocab_size=32000):
    """Create a mock tokenizer for testing."""
    # Create a dictionary mapping token IDs to strings
    vocab = {i: f"TOKEN_{i}" for i in range(vocab_size)}
    
    # Special tokens
    vocab[0] = "[PAD]"
    vocab[1] = "[BOS]"
    vocab[2] = "[EOS]"
    
    # Add some common words for more realistic output
    common_words = [
        "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
        "he", "was", "for", "on", "are", "with", "as", "I", "his", "they",
        "be", "at", "one", "have", "this", "from", "or", "had", "by", "not",
        "word", "but", "what", "some", "we", "can", "out", "other", "were", "all",
        "there", "when", "up", "use", "your", "how", "said", "an", "each", "she"
    ]
    
    for i, word in enumerate(common_words):
        vocab[i + 100] = word
    
    # Create a mock tokenizer class
    class MockTokenizer:
        def __init__(self):
            self.vocab = vocab
            self.pad_token = "[PAD]"
            self.pad_token_id = 0
            self.bos_token = "[BOS]"
            self.bos_token_id = 1
            self.eos_token = "[EOS]"
            self.eos_token_id = 2
        
        def encode(self, text, **kwargs):
            """Mock encode method that returns a list of token IDs."""
            # Split the text into words and map to token IDs
            words = text.lower().split()
            
            # Try to map words to tokens, use random tokens for unknown words
            token_ids = []
            for word in words:
                # Find the token ID for this word if it exists
                for token_id, token in self.vocab.items():
                    if token == word:
                        token_ids.append(token_id)
                        break
                else:
                    # Use a random token ID for unknown words
                    token_ids.append(100 + (hash(word) % 50))
            
            return token_ids
        
        def decode(self, token_ids, skip_special_tokens=False, **kwargs):
            """Mock decode method that returns a string."""
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            # Convert token IDs to strings
            tokens = []
            for token_id in token_ids:
                if skip_special_tokens and token_id < 100:
                    continue
                tokens.append(self.vocab.get(token_id, f"UNK_{token_id}"))
            
            # Join tokens into a string
            return " ".join(tokens)
        
        def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
            """Mock batch_decode method that returns a list of strings."""
            return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]
        
        def __call__(self, texts, return_tensors=None, padding=False, **kwargs):
            """Mock __call__ method that tokenizes text and returns tensors."""
            if isinstance(texts, str):
                texts = [texts]
            
            # Encode each text
            encoded = [self.encode(text) for text in texts]
            
            # Pad sequences if requested
            if padding:
                max_len = max(len(seq) for seq in encoded)
                encoded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
            
            # Convert to tensors if requested
            if return_tensors == "pt":
                input_ids = torch.tensor(encoded)
                attention_mask = torch.ones_like(input_ids)
                if padding:
                    attention_mask = torch.where(input_ids != 0, 1, 0)
                
                # Create a class with a proper to method
                class TokenizerOutput:
                    def __init__(self, input_ids, attention_mask):
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                    
                    def to(self, device):
                        self.input_ids = self.input_ids.to(device)
                        self.attention_mask = self.attention_mask.to(device)
                        return self
                
                return TokenizerOutput(input_ids, attention_mask)
            
            return encoded
    
    return MockTokenizer()


def patch_enhanced_inference():
    """Patch the EnhancedInference class to use our mock model."""
    # Save the original methods
    original_load_model = EnhancedInference._load_model
    original_generate_stream = EnhancedInference._generate_stream
    
    # Define a new _load_model method that uses our mock model
    def mock_load_model(self):
        print(f"Loading mock model for testing")
        self.model = MockEnhancedDeepShivaMoE()
        self.tokenizer = create_mock_tokenizer()
        self.model.eval()
    
    # Define a simplified _generate_stream method for testing
    def mock_generate_stream(self, inputs, max_new_tokens, temperature, top_p, top_k, 
                           repetition_penalty, do_sample, stop_strings, callback):
        # Generate a simple response token by token
        tokens = ["Once ", "upon ", "a ", "time ", "in ", "a ", "land ", "far ", "away, ",
                 "there ", "lived ", "a ", "brave ", "knight ", "who ", "fought ", "dragons."]
        
        for token in tokens:
            yield token
            if callback:
                callback(token)
    
    # Replace the methods
    EnhancedInference._load_model = mock_load_model
    EnhancedInference._generate_stream = mock_generate_stream
    
    return original_load_model, original_generate_stream


def test_text_generation():
    """Test text generation with the enhanced inference API."""
    print("\n=== Testing Text Generation ===")
    
    # Create an instance of the enhanced inference engine
    inference = EnhancedInference(
        model_path="mock_model",
        device="cpu",
        precision="fp32",
    )
    
    # Generate text
    prompt = "Once upon a time in a land far away"
    print(f"Prompt: {prompt}")
    
    print("\nStandard generation:")
    output = inference.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.7,
        stream=False,
    )
    print(f"Output: {output}")
    
    print("\nStreaming generation:")
    print("Output: ", end="", flush=True)
    for token in inference.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.7,
        stream=True,
    ):
        print(token, end="", flush=True)
    print()


def test_code_completion():
    """Test code completion with the enhanced inference API."""
    print("\n=== Testing Code Completion ===")
    
    # Create an instance of the enhanced inference engine
    inference = EnhancedInference(
        model_path="mock_model",
        device="cpu",
        precision="fp32",
    )
    
    # Complete code
    code_prompt = "def fibonacci(n):\n    "
    print(f"Code Prompt: {code_prompt}")
    
    output = inference.complete_code(
        code_prompt=code_prompt,
        language="python",
        max_new_tokens=30,
        temperature=0.2,
    )
    print(f"Completed Code: {output}")


def test_math_solving():
    """Test math problem solving with the enhanced inference API."""
    print("\n=== Testing Math Problem Solving ===")
    
    # Create an instance of the enhanced inference engine
    inference = EnhancedInference(
        model_path="mock_model",
        device="cpu",
        precision="fp32",
    )
    
    # Solve math problem
    problem = "Solve the equation: 2x + 5 = 15"
    print(f"Problem: {problem}")
    
    solution = inference.solve_math(
        problem=problem,
        max_new_tokens=50,
        temperature=0.3,
        show_work=True,
    )
    print(f"Solution: {solution}")


def test_translation():
    """Test translation with the enhanced inference API."""
    print("\n=== Testing Translation ===")
    
    # Create an instance of the enhanced inference engine
    inference = EnhancedInference(
        model_path="mock_model",
        device="cpu",
        precision="fp32",
    )
    
    # Translate text
    text = "Hello, how are you doing today?"
    source_lang = "English"
    target_lang = "Spanish"
    print(f"Text: {text}")
    print(f"Translating from {source_lang} to {target_lang}")
    
    translation = inference.translate(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        max_new_tokens=30,
        temperature=0.5,
    )
    print(f"Translation: {translation}")


def test_batch_processing():
    """Test batch processing with the enhanced inference API."""
    print("\n=== Testing Batch Processing ===")
    
    # Create an instance of the enhanced inference engine
    inference = EnhancedInference(
        model_path="mock_model",
        device="cpu",
        precision="fp32",
        batch_size=2,
    )
    
    # Process multiple prompts
    prompts = [
        "Tell me about artificial intelligence",
        "What is the capital of France?",
    ]
    print(f"Prompts: {prompts}")
    
    outputs = inference.batch_process(
        prompts=prompts,
        max_new_tokens=20,
        temperature=0.7,
    )
    
    for i, output in enumerate(outputs):
        print(f"Output {i+1}: {output}")


def main():
    """Run all tests."""
    # Patch the EnhancedInference class
    original_load_model, original_generate_stream = patch_enhanced_inference()
    
    try:
        # Run tests
        test_text_generation()
        test_code_completion()
        test_math_solving()
        test_translation()
        test_batch_processing()
        
        print("\n=== All tests completed successfully ===")
    finally:
        # Restore the original methods
        EnhancedInference._load_model = original_load_model
        EnhancedInference._generate_stream = original_generate_stream


if __name__ == "__main__":
    main()
