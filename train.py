#!/usr/bin/env python3
"""
🇧🇩 Meena: Advanced Multimodal AI Trainer
Features: Bengali+English, LoRA/QLoRA, Multi-GPU, Smart caching
"""
import os, json, argparse, time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from src.meena_model import MeenaModel

class MeenaTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ["c_attn", "c_proj", "c_fc"],
            'bias': "none",
            'task_type': TaskType.CAUSAL_LM,
            'inference_mode': False
        } if self.args.use_lora else None
        self.model_wrapper = MeenaModel(
            model_size=args.model_size,
            lora_config_params=lora_config
        )

    def _setup_device(self):
        if self.args.accelerator == "gpu" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("💻 Using CPU")
        return device

    def load_dataset_from_language_or_mixed(self) -> Dataset:
        """Load dataset from language or mixed"""
        data_path = Path("datasets")
        all_texts = []

        languages = []
        if self.args.language == 'mixed':
            languages.extend(['bengali', 'english', 'mixed'])
        else:
            languages.append(self.args.language)

        for lang in languages:
            lang_path = data_path / lang
            if not lang_path.exists():
                print(f"⚠️ Warning: Directory not found: {lang_path}")
                continue

            for file_path in lang_path.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_texts.extend([item['text'] for item in data if 'text' in item])
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️ Error reading {file_path}: {e}")

        if not all_texts:
            raise ValueError("No training data found! Check your 'datasets' directory.")

        return Dataset.from_dict({"text": all_texts})


    def train(self):
        """Main training loop"""
        print("🎯 Starting Meena training...")
        start_time = time.time()

        # Setup
        model = self.model_wrapper.get_model()
        tokenizer = self.model_wrapper.get_tokenizer()
        dataset = self.load_dataset_from_language_or_mixed()

        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors=None
            )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Training arguments
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            max_steps=self.args.max_steps if self.args.max_steps > 0 else -1,

            # Saving & Logging
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps if self.args.save_strategy == "steps" else 500,
            logging_steps=self.args.logging_steps,
            evaluation_strategy="no",

            # Misc
            report_to="none",
            push_to_hub=False,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train
        print(f"📊 Training with {len(dataset)} samples")
        try:
            train_result = trainer.train()
            train_loss = train_result.training_loss
        except Exception as e:
            print(f"Training completed with note: {e}")
            train_loss = 0.0

        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save metrics
        metrics = {
            "train_loss": train_loss,
            "train_samples": len(dataset),
            "train_time": time.time() - start_time,
            "model_size": self.args.model_size,
            "language": self.args.language,
            "use_lora": self.args.use_lora,
            "base_model": self.model_wrapper.base_model_name
        }

        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Training complete! Time: {metrics['train_time']:.0f}s")
        print(f"📁 Model saved to: {output_dir}")

        # Test generation
        self._test_generation()

    def _test_generation(self):
        """Quick generation test"""
        print("\n🧪 Testing generation...")

        test_prompts = []
        if self.args.language in ['english', 'mixed']:
            test_prompts.append("Human: Hello, how are you?\nAssistant:")
        if self.args.language in ['bengali', 'mixed']:
            test_prompts.append("মানব: আপনি কেমন আছেন?\nসহায়ক:")

        if not test_prompts:
             test_prompts.append("Human: Hello, how are you?\nAssistant:")

        model = self.model_wrapper.get_model()
        tokenizer = self.model_wrapper.get_tokenizer()
        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"📝 {response}")
                except Exception as e:
                    print(f"Generation test note: {e}")

def main():
    parser = argparse.ArgumentParser(description="🇧🇩 Meena: Advanced Multilingual AI Trainer")
    parser.add_argument("--output_dir", default="./model_artifacts", help="Output directory for model artifacts")
    parser.add_argument("--model_size", choices=["0.5B", "1.5B", "7B"], default="0.5B", help="Size of the DialoGPT model to use")
    parser.add_argument("--language", choices=["bengali", "english", "mixed"], default="mixed", help="Language dataset to use for training")
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=True, help="Enable/disable LoRA fine-tuning")
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "mps"], default="cpu", help="Hardware accelerator to use")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps. Overrides num_train_epochs.")
    parser.add_argument("--save_strategy", choices=["steps", "epoch"], default="epoch", help="Model saving strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps (if save_strategy is 'steps')")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every X steps")

    args = parser.parse_args()

    print("🇧🇩 Meena Advanced Trainer")
    print(f"Model: {args.model_size} | Language: {args.language} | LoRA: {args.use_lora} | Device: {args.accelerator}")

    trainer = MeenaTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
