from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Create the directory if it doesn't exist
os.makedirs("models/pretrained", exist_ok=True)

# Optionally login with your token (you can skip if already logged in via CLI)
# login(token="your_huggingface_token_here")

# Download the model and tokenizer
model_name = "codellama/CodeLlama-7b-hf"
print(f"Downloading {model_name}...")

# This will download the model to the cache
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    low_cpu_mem_usage=True
)

# Save to your specified location
save_path = "models/pretrained/codellama-7b"
print(f"Saving model to {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Download complete!")