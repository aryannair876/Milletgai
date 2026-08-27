"""
DPO-Only Training Script for MilletGAI
=======================================
Run this AFTER train_sft.py completes to add preference tuning.
Loads the existing SFT model and applies DPO for better alignment.

Memory-optimized for RTX 4060 8GB.
"""

import os
import gc
import sys
import json
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['USE_TRITON'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch

# TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Windows fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset

print("=" * 60)
print("DPO TRAINING - MilletGAI Phase 2")
print("Direct Preference Optimization")
print("=" * 60)

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "Meta-Llama-3-8B-Instruct"
SFT_MODEL_PATH = "models/milletsgai-final"  # Load the SFT model produced by train_sft.py

# Preference pairs: CoT responses as "chosen", diverse rejection types as "rejected".
DPO_DATASET_PATH = "data/millets_dpo_final.jsonl"

OUTPUT_DIR = "models/milletsgai-dpo"

# ULTRA-SAFE SETTINGS - THE ONLY CONFIG THAT WORKED WITHOUT CRASHING
# Tested: 500 examples completed in 14 min, VRAM peaked at 7.16GB
# Higher settings cause PC restart - hardware limitation
MAX_SEQ_LENGTH = 128  # Back to safe value - 256 caused VRAM crash
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
DPO_LEARNING_RATE = 5e-5
DPO_BETA = 0.1
LORA_R = 16
LORA_ALPHA = 32
COOLING_PAUSE_EVERY = 5  # More frequent cooling to prevent crash

# =============================================================================
# MEMORY HELPERS
# =============================================================================

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

import time
def thermal_pause(step):
    """Pause periodically to let GPU cool down"""
    if step > 0 and step % COOLING_PAUSE_EVERY == 0:
        print(f"   💨 Cooling pause at step {step}...")
        clean_memory()
        time.sleep(3)  # 3 second pause to let GPU temperature drop

def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        return f"{allocated:.2f}GB"
    return "No GPU"

# =============================================================================
# CREATE DPO DATASET
# =============================================================================

def create_dpo_dataset():
    """Create preference pairs for DPO training"""
    print("\n📝 Creating DPO preference dataset...")
    
    dpo_path = Path(DPO_DATASET_PATH)
    
    # Check if already exists
    if dpo_path.exists():
        print(f"   ✅ Loading: {dpo_path}")
        dataset = load_dataset('json', data_files=str(dpo_path), split='train')
        # Use full dataset with safe MAX_SEQ_LENGTH=128
        print(f"   📊 Using all {len(dataset)} examples")
        return dataset
    
    # Load SFT dataset to create pairs
    sft_path = "data/millets_data_final.jsonl"
    with open(sft_path, 'r', encoding='utf-8') as f:
        sft_data = [json.loads(line) for line in f]
    
    dpo_data = []
    
    for example in sft_data:
        instruction = example.get('instruction', '').lower()
        correct = example.get('output', '')
        
        # Create rejection based on common errors
        rejected = None
        
        if 'chena' in instruction:
            rejected = "Chena is a type of fresh Indian cheese made from curdled milk. It's used in sweets like rasgulla."
        elif 'ph' in instruction or 'soil' in instruction:
            rejected = "The optimal soil pH for all millets is 6.0-7.0, which provides ideal growing conditions."
        elif 'protein' in instruction:
            rejected = correct.replace("12.5", "8.0").replace("11.6", "7.0").replace("10.4", "6.0") if any(x in correct for x in ["12.5", "11.6", "10.4"]) else None
        elif 'compare' in instruction:
            rejected = "Both millets have similar nutritional profiles and can be used interchangeably."
        
        if rejected and rejected != correct:
            dpo_data.append({
                "prompt": example['instruction'],
                "chosen": correct,
                "rejected": rejected,
            })
    
    # Save
    dpo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dpo_path, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Created {len(dpo_data)} preference pairs")
    return Dataset.from_list(dpo_data)

# =============================================================================
# LOAD MODEL
# =============================================================================

def load_sft_model():
    """Load the SFT model with LoRA adapters"""
    print(f"\n🔧 Loading SFT model from: {SFT_MODEL_PATH}")
    
    clean_memory()
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load base model
    print("   Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load LoRA adapters from SFT training
    print("   Loading LoRA adapters...")
    if Path(SFT_MODEL_PATH).exists():
        model = PeftModel.from_pretrained(
            base_model,
            SFT_MODEL_PATH,
            is_trainable=True,
        )
        print(f"   ✅ Loaded SFT adapters from {SFT_MODEL_PATH}")
    else:
        print(f"   ⚠️ SFT model not found, creating new LoRA...")
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model)
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    print(f"   ✅ Model ready! VRAM: {get_gpu_memory()}")
    model.print_trainable_parameters()
    
    return model, tokenizer

# =============================================================================
# DPO TRAINING
# =============================================================================

def run_dpo(model, tokenizer, dpo_dataset):
    """Run DPO training"""
    # LAPTOP-SAFE: Clear all memory before starting
    clean_memory()
    
    print(f"\n{'='*60}")
    print("🎯 Starting DPO Training (LAPTOP-SAFE MODE)")
    print(f"{'='*60}")
    print(f"   Examples: {len(dpo_dataset)}")
    print(f"   VRAM before training: {get_gpu_memory()}")
    print(f"   Beta: {DPO_BETA}")
    print(f"   Learning Rate: {DPO_LEARNING_RATE}")
    
    # DPO config - memory optimized for 8GB VRAM
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=DPO_BETA,
        learning_rate=DPO_LEARNING_RATE,
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH // 2,
        # Memory fix: precompute ref logprobs once instead of keeping ref model in VRAM
        precompute_ref_log_probs=True,
        # LAPTOP-SAFE: Reduce memory pressure
        dataloader_pin_memory=False,  # Prevents memory spikes
        dataloader_num_workers=0,  # Single-threaded to avoid memory duplication
    )
    
    # ULTRA-SAFE: Custom callback for VRAM protection
    from transformers import TrainerCallback
    
    class VRAMSafetyCallback(TrainerCallback):
        """Monitor VRAM and clear memory if getting close to limit"""
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / 1024**3
                vram_max = torch.cuda.max_memory_allocated() / 1024**3
                
                # Log VRAM usage
                if state.global_step % 5 == 0:
                    print(f"   📊 Step {state.global_step}: VRAM {vram_gb:.2f}GB (peak: {vram_max:.2f}GB)")
                
                # Emergency clear if VRAM > 6.5GB (approaching 8GB limit)
                if vram_gb > 6.5:
                    print(f"   ⚠️ VRAM critical ({vram_gb:.2f}GB)! Emergency clear...")
                    clean_memory()
                    time.sleep(1)
                
                # Regular pause every 10 steps
                if state.global_step > 0 and state.global_step % COOLING_PAUSE_EVERY == 0:
                    clean_memory()
                    time.sleep(2)
                    
            return control
    
    # DPO trainer with thermal safety
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will create a copy
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        args=dpo_config,
        callbacks=[VRAMSafetyCallback()],  # VRAM protection
    )
    
    print("\n🚀 Training (ULTRA-SAFE MODE with cooling pauses)...\n")
    trainer.train()
    
    # Cleanup
    del trainer
    clean_memory()
    
    return model

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n🌾 MilletGAI DPO Training Pipeline")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # Step 1: Create/load DPO dataset
        dpo_dataset = create_dpo_dataset()
        
        # Step 2: Load SFT model
        model, tokenizer = load_sft_model()
        
        # Step 3: Run DPO
        model = run_dpo(model, tokenizer, dpo_dataset)
        
        # Step 4: Save final model
        print(f"\n💾 Saving final model to: {OUTPUT_DIR}")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"\n✅ SUCCESS! DPO training complete!")
        print(f"   Model saved to: {OUTPUT_DIR}")
        print(f"\nNext steps:")
        print(f"   python ingest_data.py          # build the RAG knowledge base")
        print(f"   python rag_inference.py \"...\"  # query the model directly")
        print(f"   python backend_api.py          # serve the API on :8000")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except torch.cuda.OutOfMemoryError:
        print("\n\n❌ OUT OF MEMORY!")
        print("Try reducing MAX_SEQ_LENGTH to 384")
        clean_memory()
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
