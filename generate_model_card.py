#!/usr/bin/env python3
"""
Generate comprehensive model card for Meena
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

def generate_model_card(model_path: str, output_file: str, metrics_file: str = None):
    """Generate a comprehensive model card"""

    # Load metrics if available
    metrics = {}
    if metrics_file and Path(metrics_file).exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except:
            pass

    # Load training metrics if available
    training_metrics = {}
    training_metrics_path = Path(model_path) / "training_metrics.json"
    if training_metrics_path.exists():
        try:
            with open(training_metrics_path) as f:
                training_metrics = json.load(f)
        except:
            pass

    model_card = f"""---
license: apache-2.0
pipeline_tag: text-generation
language:
- en
- bn
tags:
- meena
- bangladesh
- multilingual
- conversational-ai
- fine-tuned
library_name: transformers
datasets:
- custom
metrics:
- perplexity
base_model: {training_metrics.get('base_model', 'microsoft/DialoGPT-small')}
---

# 🇧🇩 Meena - Multilingual Conversational AI

## Model Description

Meena is an advanced multilingual conversational AI model developed in Bangladesh. It supports both Bengali (বাংলা) and English languages, designed to provide helpful, harmless, and honest responses.

### Key Features

- 🌍 **Multilingual**: Native support for Bengali and English
- 🤖 **Conversational**: Optimized for natural dialogue
- ⚡ **Efficient**: Fine-tuned with LoRA for fast inference
- 🎯 **Specialized**: Trained on Bengali cultural context
- 🔧 **Flexible**: Supports various conversation formats

## Model Details

- **Developed by**: Bangladesh AI Community
- **Model type**: Causal Language Model
- **Language(s)**: Bengali, English
- **License**: Apache 2.0
- **Base Model**: {training_metrics.get('base_model', 'microsoft/DialoGPT-small')}
- **Finetuned from**: Pre-trained language model with LoRA adaptation

### Training Details

- **Training Data**: {training_metrics.get('train_samples', 'Custom')} conversation samples
- **Training Time**: {training_metrics.get('train_time', 'N/A')} seconds
- **Bengali Support**: {training_metrics.get('bengali_enabled', True)}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)

### Performance Metrics

{f"- **Model Size**: {metrics.get('model_size_mb', 'N/A')} MB" if metrics.get('model_size_mb') else ""}
{f"- **Parameters**: {metrics.get('total_parameters', 'N/A')} total" if metrics.get('total_parameters') else ""}
{f"- **Trainable Parameters**: {metrics.get('trainable_parameters', 'N/A')}" if metrics.get('trainable_parameters') else ""}
{f"- **Average Inference Time**: {metrics.get('avg_inference_time', 'N/A')} seconds" if metrics.get('avg_inference_time') else ""}
{f"- **Successful Generations**: {metrics.get('successful_generations', 'N/A')}" if metrics.get('successful_generations') else ""}

## Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "likhonsheikh/Meena"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# English conversation
prompt = "Human: Hello, how are you?\\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Bengali Usage

```python
# Bengali conversation
bengali_prompt = "মানব: আপনি কেমন আছেন?\\nসহায়ক:"
inputs = tokenizer(bengali_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Data

The model was trained on a carefully curated dataset containing:

- **English Conversations**: High-quality dialogue samples covering various topics
- **Bengali Conversations**: Native Bengali conversations with cultural context
- **Mixed Language**: Code-switching examples between Bengali and English
- **Safety Guidelines**: Responses aligned with helpful, harmless, honest principles

## Intended Use

### Primary Use Cases

- **Conversational AI**: Chat applications and virtual assistants
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(model_card)

    print(f"✅ Model card generated: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output file path for README.md")
    parser.add_argument("--metrics", help="Path to benchmark_results.json")
    args = parser.parse_args()

    generate_model_card(args.model_path, args.output, metrics_file=args.metrics)

if __name__ == "__main__":
    main()
