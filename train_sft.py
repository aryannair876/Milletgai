"""
Fast SFT Training Script for MilletGAI
======================================
Simplified and optimized for maximum speed on RTX 4060 8GB.
Pre-tokenizes data for faster training.
"""

import os
import gc
import sys
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['USE_TRITON'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch

# TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Windows fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

print("=" * 60)
print("FAST SFT TRAINING - MilletGAI")
print("=" * 60)

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "Meta-Llama-3-8B-Instruct"
DATASET_PATH = "data/millets_data_final.jsonl"
OUTPUT_DIR = "models/milletsgai-final"

MAX_SEQ_LENGTH = 512          # Back to 512 for speed
BATCH_SIZE = 1                 # Limited by 8GB VRAM
GRADIENT_ACCUMULATION = 16     # Larger effective batch (16) for stable gradients
LEARNING_RATE = 1.5e-4         # Slightly lower for better convergence
EPOCHS = 6                     # 6 epochs - worth it for domain learning
LORA_R = 64                    # Back to 64 for speed
LORA_ALPHA = 128               # 2x rank scaling

# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    print("\n🔧 Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# =============================================================================
# PREPARE DATASET
# =============================================================================

SYSTEM_PROMPT = """You are MilletGAI, an expert on millets in India.
RULES:
1. Chena is Proso Millet, NOT cheese
2. Never make up data
3. Only discuss millets"""

def format_example(example):
    """Format example as chat template"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""

def prepare_dataset():
    print("\n📂 Loading and preparing dataset...")
    
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    print(f"   Loaded {len(dataset)} examples")
    
    # Add text field
    dataset = dataset.map(
        lambda x: {"text": format_example(x)},
        num_proc=1,  # Single process for stability
        desc="Formatting"
    )
    
    return dataset

# =============================================================================
# TRAIN
# =============================================================================

def train():
    gc.collect()
    torch.cuda.empty_cache()
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    model, tokenizer = load_model()
    dataset = prepare_dataset()
    
    print(f"\n📊 Training Configuration:")
    print(f"   Examples: {len(dataset)}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    
    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,  # Disable packing for simplicity
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save
    print("\n💾 Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Done! Model saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  python simple_chat.py")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
