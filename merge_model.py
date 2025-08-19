# merge_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_MODEL = "./tinyllama-stock-advisor" # The folder with your fine-tuned adapter
MERGED_MODEL_DIR = "./tinyllama-stock-advisor-merged" # The output folder for the full model

def merge_and_save():
    """
    Loads the base model and the LoRA adapter, merges them,
    and saves the complete model to a new directory.
    """
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from: {ADAPTER_MODEL}")
    # Load the PEFT model (adapter) on top of the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

    print("Merging the adapter into the base model...")
    # Merge the adapter weights into the base model
    model = model.merge_and_unload()

    print(f"Saving the merged model to: {MERGED_MODEL_DIR}")
    # Create the directory if it doesn't exist
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    
    # Save the fully merged model and the tokenizer
    model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    print("\nMerge complete! Your full model is ready.")

if __name__ == "__main__":
    merge_and_save()
