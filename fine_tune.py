# fine_tune.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig # Import SFTConfig
import os

# --- Configuration ---
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
DATASET_FILE = "stock_advisor_dataset.jsonl"
NEW_MODEL_NAME = "tinyllama-stock-advisor"


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

def run_finetuning():
    """Loads the dataset, configures the model, and runs the fine-tuning process."""

    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
        print("="*50)
        print(f"ERROR: The dataset file '{DATASET_FILE}' is missing or empty.")
        print("Please run 'create_dataset.py' successfully before starting the fine-tuning.")
        print("="*50)
        return

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # **THE FIX: Use SFTConfig instead of TrainingArguments**
    # This is the modern and correct way to configure the SFTTrainer.
    training_args = SFTConfig(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,  # Use float16 for wider compatibility
        bf16=True, # Ensure bfloat16 is disabled
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        dataset_text_field="text", # SFTConfig requires this field
        report_to=[], 
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        # tokenizer=tokenizer, # SFTConfig requires the tokenizer here
        # max_seq_length=None,
        # packing=False,
    )

    print("Starting fine-tuning process...")
    trainer.train()

    print(f"Fine-tuning complete. Saving model to '{NEW_MODEL_NAME}'...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print("Model saved successfully!")

if __name__ == "__main__":
    run_finetuning()
