#!/usr/bin/env python3
"""
Meena Benchmark & Performance Testing
"""
import json
import time
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from datasets import Dataset

def load_evaluation_data(eval_lang: str) -> Dataset:
    """Load evaluation data from the specified language directory."""
    print(f"📚 Loading evaluation data for '{eval_lang}'...")
    data_path = Path("datasets") / eval_lang
    all_texts = []

    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation data directory not found: {data_path}")

    for file_path in data_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_texts.extend([item['text'] for item in data if 'text' in item])
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    if not all_texts:
        raise ValueError(f"No evaluation data found in {data_path}")

    # For demonstration, we'll split prompts and references.
    # In a real scenario, you'd have dedicated test sets.
    prompts = [t.split("\n")[0] for t in all_texts]
    references = [t.split("\n")[1] for t in all_texts]

    return Dataset.from_dict({"prompt": prompts, "reference": references})


def benchmark_model(model_path: str, output_file: str, eval_lang: str):
    """Run comprehensive benchmarks"""
    print("⚡ Starting Meena benchmarks...")

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load evaluation data
        eval_dataset = load_evaluation_data(eval_lang)

        results = {
            "model_size_mb": 0,
            "avg_inference_time_ms": 0,
            "perplexity": None,
            "bleu_score": None,
            "total_parameters": 0,
            "trainable_parameters": 0
        }

        # Model size
        try:
            model_files = list(Path(model_path).glob("pytorch_model*.bin"))
            if model_files:
                results["model_size_mb"] = sum(f.stat().st_size for f in model_files) / (1024*1024)
        except:
            pass

        # Parameter count
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results["total_parameters"] = total_params
            results["trainable_parameters"] = trainable_params
        except:
            pass

        # --- NLP Metric Calculations ---
        print("\n📈 Calculating NLP metrics...")
        model.eval()

        # 1. Perplexity
        try:
            print("  - Calculating Perplexity...")
            perplexity_metric = evaluate.load("perplexity", module_type="metric")
            # Join prompt and reference for perplexity calculation
            full_texts = [p + "\n" + r for p, r in zip(eval_dataset["prompt"], eval_dataset["reference"])]
            ppl_results = perplexity_metric.compute(predictions=full_texts, model_id=model_path)
            results["perplexity"] = ppl_results['mean_perplexity']
            print(f"  ✅ Perplexity: {results['perplexity']:.2f}")
        except Exception as e:
            print(f"  ⚠️ Could not calculate Perplexity: {e}")

        # 2. BLEU Score & Inference Time
        print("  - Calculating BLEU score and Inference Time...")
        bleu_metric = evaluate.load("bleu")
        inference_times = []
        predictions = []

        with torch.no_grad():
            for i, item in enumerate(eval_dataset):
                try:
                    start_time = time.time()
                    inputs = tokenizer(item["prompt"], return_tensors="pt", truncation=True, max_length=128)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    end_time = time.time()
                    inference_times.append(end_time - start_time)

                    # Extract only the generated part
                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    predictions.append(response)
                    if i < 5: # Print first 5 generations
                        print(f"    - Generated: {response[:100]}...")

                except Exception as e:
                    print(f"    ⚠️ Generation failed for prompt: {item['prompt']}")

        try:
            bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in eval_dataset["reference"]])
            results["bleu_score"] = bleu_results['bleu'] * 100  # As a percentage
            print(f"  ✅ BLEU Score: {results['bleu_score']:.2f}")
        except Exception as e:
            print(f"  ⚠️ Could not calculate BLEU score: {e}")


        results["avg_inference_time_ms"] = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print("📊 Benchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        print(f"✅ Benchmark complete! Results saved to {output_file}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        # Save empty results
        with open(output_file, "w") as f:
            json.dump({"error": str(e), "status": "failed"}, f)

def main():
    parser = argparse.ArgumentParser(description="Meena Benchmark & Performance Testing")
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--output_file", required=True, help="Path to save the output benchmark JSON file")
    parser.add_argument("--eval_lang", choices=["bengali", "english", "mixed"], default="english", help="Language of the evaluation dataset to use")
    args = parser.parse_args()

    benchmark_model(args.model_path, args.output_file, args.eval_lang)

if __name__ == "__main__":
    main()
