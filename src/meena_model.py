import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

class MeenaModel:
    def __init__(self, model_size="0.5B", lora_config_params=None):
        self.model_configs = {
            "0.5B": "microsoft/DialoGPT-small",
            "1.5B": "microsoft/DialoGPT-medium",
            "7B": "microsoft/DialoGPT-large"
        }
        if model_size not in self.model_configs:
            raise ValueError(f"Invalid model size. Choose from {list(self.model_configs.keys())}")

        self.base_model_name = self.model_configs[model_size]
        self.lora_config_params = lora_config_params if lora_config_params else {}

        self.model = None
        self.tokenizer = None

    def setup_model_and_tokenizer(self):
        """Setup model with tokenizer"""
        print(f"🚀 Loading {self.base_model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        return self.model, self.tokenizer

    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        if not self.model:
            raise ValueError("Model must be loaded before setting up LoRA.")

        # Default LoRA parameters if not provided
        default_lora_params = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ["c_attn", "c_proj", "c_fc"],
            'bias': "none",
            'task_type': TaskType.CAUSAL_LM,
            'inference_mode': False
        }

        # Override defaults with any user-provided params
        config_params = {**default_lora_params, **self.lora_config_params}

        lora_config = LoraConfig(**config_params)

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self.model

    def get_model(self):
        """Returns the configured model, ready for training."""
        if not self.model:
            self.setup_model_and_tokenizer()
            self.setup_lora()
        return self.model

    def get_tokenizer(self):
        """Returns the configured tokenizer."""
        if not self.tokenizer:
            self.setup_model_and_tokenizer()
        return self.tokenizer
