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

def benchmark_model(model_path: str, output_file: str):
    """Run comprehensive benchmarks"""
    print("⚡ Starting Meena benchmarks...")

    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test prompts
        test_prompts = [
            "Human: Hello, how are you?\nAssistant:",
            "Human: What is artificial intelligence?\nAssistant:",
            "Human: Tell me a short story.\nAssistant:",
        ]

        results = {
            "model_size_mb": 0,
            "avg_inference_time": 0,
            "successful_generations": 0,
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

        # Inference benchmarks
        model.eval()
        inference_times = []
        successful_gens = 0

        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    start_time = time.time()
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                    successful_gens += 1

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"✅ Generated: {response[:100]}...")

                except Exception as e:
                    print(f"⚠️ Generation failed: {e}")

        results["avg_inference_time"] = sum(inference_times) / len(inference_times) if inference_times else 0
        results["successful_generations"] = successful_gens

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    args = parser.parse_args()

    benchmark_model(args.model_path, args.output_file)

if __name__ == "__main__":
    main()
