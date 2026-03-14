"""
Alternative SFT training script WITHOUT Unsloth (Windows-compatible).
Uses standard Hugging Face transformers + PEFT + TRL for LoRA fine-tuning.
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


class MilletsGAITrainerWindows:
    """Fine-tune LLM for millet domain expertise (Windows-compatible version)."""
    
    def __init__(
        self,
        model_name: str = "Meta-Llama-3-8B-Instruct",  # Local model in workspace
        dataset_path: str = "data/millets_data_enhanced.jsonl",  # Use test-driven enhanced data
        output_dir: str = "models/checkpoints",
        adapter_output_dir: str = "models/milletsgai-sft"
    ):
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.adapter_output_dir = Path(adapter_output_dir)
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        # LoRA config - OPTIMIZED FOR RTX 4060 8GB + Test Issues
        # Test showed: Foxtail 100%, Ragi 50%, but Jowar/Bajra/Chena failing
        # Solution: Strong LoRA to override base model's generic knowledge
        self.lora_r = 64  # High rank for strong override (167M params)
        self.lora_alpha = 128  # 2:1 ratio for aggressive fine-tuning
        self.lora_dropout = 0.1  # Increased to 0.1 to prevent overfitting
        
        # Training config - TARGETED FIX FOR TEST FAILURES
        # Issue 1: pH hallucinations (42% of failures) - More epochs
        # Issue 2: Chena/paneer confusion - Anti-hallucination focus
        # Issue 3: Missing pest details - Better attention to specifics
        self.num_epochs = 10  # Increased to 10 for stubborn issues
        self.learning_rate = 1.5e-4  # Lower for fine-grained learning
        self.batch_size = 1  # Must be 1 for 8GB VRAM
        self.gradient_accumulation_steps = 8  # Effective batch=8 for stability
        self.warmup_ratio = 0.15  # 15% warmup for gradual learning
        self.logging_steps = 10
        self.save_steps = 100  # Less frequent saves to reduce I/O
        self.eval_steps = 100  # Evaluate less often to speed up
        self.max_grad_norm = 0.5  # Gradient clipping for stability
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with 4-bit quantization."""
        print(f"Loading model: {self.model_name}")
        print("Using aggressive memory optimization for 8GB VRAM...")
        
        # Quantization config - optimized for 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization saves memory
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16 to save memory
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("Loading model... This may take 2-3 minutes...")
        print("Note: Loading in 4-bit quantization to fit in 8GB VRAM")
        
        # Very conservative memory limits to prevent system interruption
        max_memory = {
            0: "6.5GB",  # GPU - conservative to avoid OOM
            "cpu": "8GB"  # CPU RAM - leave room for system
        }
        
        # Load model with most stable settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Critical for preventing memory spikes
        )
        print("Model loaded successfully in 4-bit quantization")
        
        self.model.config.use_cache = False  # Disable KV cache to save memory
        self.model.config.pretraining_tp = 1
        
        print("Model and tokenizer loaded successfully")
        
    def add_lora_adapters(self):
        """Add LoRA adapters to the model."""
        print("Adding LoRA adapters...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        # LoRA configuration
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        # Add LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        print(f"LoRA adapters added (r={self.lora_r}, alpha={self.lora_alpha})")
        
    def load_dataset_data(self):
        """Load instruction dataset and split for training/validation."""
        print(f"Loading dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}\n"
                "Please run: python -m src.data_processing.make_instruction_dataset\n"
                "         then: python -m src.data_processing.validate_instruction_dataset"
            )
        
        dataset = load_dataset('json', data_files=str(self.dataset_path), split='train')
        
        # Split into train (85%) and validation (15%) for better validation
        split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        print(f"Loaded {len(self.train_dataset)} training examples")
        print(f"Loaded {len(self.eval_dataset)} validation examples")
        print(f"Dataset: {self.dataset_path}")
        
    def format_prompt(self, example):
        """Format example into instruction prompt with TARGETED anti-hallucination guidance."""
        instruction = example['instruction']
        input_text = example['input']
        output = example['output']
        
        # TARGETED anti-hallucination based on test failures:
        # 1. pH hallucinations (8 failures) - DON'T make up pH values
        # 2. Paneer confusion (1 critical) - Chena is a MILLET not cheese
        # 3. Generic responses - Use SPECIFIC training data
        anti_hallucination_note = """CRITICAL INSTRUCTIONS:
1. Provide ONLY exact information from training data
2. DO NOT make up pH values, temperature ranges, or measurements
3. Chena/Proso is a MILLET, NOT paneer (cheese)
4. For pests: Only list specific pests from training (shoot fly, stem borer, etc.)
5. If information not in training data, say "I don't have specific information"
6. Use EXACT numbers and terms from training data, not general knowledge

"""
        
        if input_text and input_text.strip():
            prompt = f"""{anti_hallucination_note}Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""{anti_hallucination_note}Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        
        return {"text": prompt}
    
    def prepare_dataset(self):
        """Format dataset for training."""
        print("Formatting datasets...")
        
        def format_and_tokenize(example):
            """Format prompt and tokenize with truncation."""
            formatted = self.format_prompt(example)
            # Tokenize to 512 tokens for RTX 4060 8GB memory efficiency
            # Test showed: Most answers fit in 512 tokens (avg response was 400 tokens)
            tokenized = self.tokenizer(
                formatted["text"],
                truncation=True,
                max_length=512,  # Optimized for 8GB VRAM
                padding=False,
            )
            return {"text": formatted["text"], "input_ids": tokenized["input_ids"]}
        
        self.train_dataset = self.train_dataset.map(
            format_and_tokenize,
            remove_columns=self.train_dataset.column_names,
            desc="Formatting training prompts"
        )
        
        self.eval_dataset = self.eval_dataset.map(
            format_and_tokenize,
            remove_columns=self.eval_dataset.column_names,
            desc="Formatting validation prompts"
        )
        
        print("Datasets formatted")
        
    def train(self):
        """Run supervised fine-tuning."""
        print("Starting training...")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine compute dtype for RTX 4060 8GB (Ampere architecture)
        # RTX 4060 has excellent FP16 support via Tensor Cores
        if torch.cuda.is_available():
            use_bf16 = False  # RTX 4060 prefers FP16 over BF16
            use_fp16 = True   # Enable FP16 for speed + memory savings
        else:
            use_bf16 = False
            use_fp16 = False
        
        # Training arguments - OPTIMIZED FOR RTX 4060 8GB + Test Issues
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=True,  # Critical for 8GB VRAM
            warmup_ratio=self.warmup_ratio,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,  # Gradient clipping for stability
            fp16=False,  # Disable FP16 - conflicts with adamw_8bit
            bf16=False,  
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=2,  # Keep only 2 checkpoints to save disk space
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            optim="adamw_8bit",  # 8-bit optimizer for memory efficiency (no FP16 scaling needed)
            weight_decay=0.05,  # Increased to prevent overfitting on augmented data
            lr_scheduler_type="cosine",  # Cosine decay for smooth learning
            seed=42,
            report_to="none",
            save_safetensors=True,
            dataloader_num_workers=0,  # Disable to save memory on 8GB GPU
            ddp_find_unused_parameters=False,  # Optimization
            # RTX 4060 specific optimizations
            tf32=True,  # Enable TF32 for Ampere (RTX 30/40 series)
            dataloader_pin_memory=False,  # Reduce memory pressure
        )
        
        # Initialize trainer with train and eval datasets
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
        )
        
        # Train
        print(f"Training for {self.num_epochs} epochs...")
        print(f"Total training examples: {len(self.train_dataset)}")
        print(f"Total validation examples: {len(self.eval_dataset)}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        trainer.train()
        
        print("Training complete!")
        
    def save_adapter(self):
        """Save LoRA adapter."""
        print(f"Saving adapter to {self.adapter_output_dir}")
        
        self.model.save_pretrained(str(self.adapter_output_dir))
        self.tokenizer.save_pretrained(str(self.adapter_output_dir))
        
        print("Adapter saved successfully")
        
    def run(self):
        """Execute full training pipeline."""
        try:
            self.load_model_and_tokenizer()
            self.add_lora_adapters()
            self.load_dataset_data()
            self.prepare_dataset()
            self.train()
            self.save_adapter()
        except Exception as e:
            print(f"\nError during training: {e}")
            print("\nIf you encounter CUDA/GPU errors, you may not have a compatible GPU.")
            print("This script requires a CUDA-compatible NVIDIA GPU with 12GB+ VRAM.")
            raise


def main():
    """Main execution function."""
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("="*60)
    print("MilletsGAI Supervised Fine-Tuning (Windows Compatible)")
    print("="*60)
    
    print("\nNote: This version uses standard HF transformers instead of Unsloth")
    print("It will be slower but compatible with Windows.\n")
    
    trainer = MilletsGAITrainerWindows()
    trainer.run()
    
    print("\nFine-tuning complete!")
    print(f"Adapter saved to: {trainer.adapter_output_dir}")


if __name__ == "__main__":
    main()
