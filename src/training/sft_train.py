"""
Supervised Fine-Tuning script for MilletsGAI using Unsloth + TRL + LoRA.
Fine-tunes LLaMA-3-8B-Instruct on millet instruction dataset.
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


class MilletsGAITrainer:
    """Fine-tune LLM for millet domain expertise."""
    
    def __init__(
        self,
        model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        dataset_path: str = "data/millets_data_cleaned.jsonl",
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
        
        # LoRA config
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        
        # Training config
        self.learning_rate = 2e-4
        self.batch_size = 2
        self.gradient_accumulation_steps = 8
        self.max_steps = 100  # Smoke test; increase for full training
        self.warmup_steps = 10
        self.logging_steps = 10
        self.save_steps = 50
        
    def load_model(self):
        """Load model and tokenizer with 4-bit quantization."""
        print(f"Loading model: {self.model_name}")
        
        max_seq_length = 2048
        dtype = None  # Auto-detect
        load_in_4bit = True
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        print("Model loaded successfully")
        
    def add_lora_adapters(self):
        """Add LoRA adapters to the model."""
        print("Adding LoRA adapters...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        print(f"LoRA adapters added (r={self.lora_r}, alpha={self.lora_alpha})")
        
    def load_dataset_data(self):
        """Load instruction dataset."""
        print(f"Loading dataset from {self.dataset_path}")
        
        self.dataset = load_dataset('json', data_files=str(self.dataset_path), split='train')
        
        print(f"Loaded {len(self.dataset)} examples")
        
    def format_prompt(self, example):
        """Format example into instruction prompt."""
        instruction = example['instruction']
        input_text = example['input']
        output = example['output']
        
        # Alpaca-style prompt format
        if input_text and input_text.strip():
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        
        return {"text": prompt}
    
    def prepare_dataset(self):
        """Format dataset for training."""
        print("Formatting dataset...")
        
        self.dataset = self.dataset.map(
            self.format_prompt,
            remove_columns=self.dataset.column_names,
            desc="Formatting prompts"
        )
        
        print("Dataset formatted")
        
    def train(self):
        """Run supervised fine-tuning."""
        print("Starting training...")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine compute dtype
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
                use_bf16 = True
                use_fp16 = False
            else:
                compute_dtype = torch.float16
                use_bf16 = False
                use_fp16 = True
        else:
            compute_dtype = torch.float32
            use_bf16 = False
            use_fp16 = False
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",  # Disable wandb, tensorboard
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args,
            packing=False,  # More stable for instruction tuning
        )
        
        # Train
        print(f"Training for {self.max_steps} steps...")
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
        self.load_model()
        self.add_lora_adapters()
        self.load_dataset_data()
        self.prepare_dataset()
        self.train()
        self.save_adapter()


def main():
    """Main execution function."""
    print("="*60)
    print("MilletsGAI Supervised Fine-Tuning")
    print("="*60)
    
    trainer = MilletsGAITrainer()
    trainer.run()
    
    print("\n✅ Fine-tuning complete!")
    print(f"Adapter saved to: {trainer.adapter_output_dir}")


if __name__ == "__main__":
    main()
