"""LLM model loading and caching utilities."""

from typing import Optional, Tuple, Any


class LLMModelCache:
    """Singleton cache for LLM models to avoid repeated loading.

    This class implements a singleton pattern to ensure that LLM models
    are loaded only once and reused across multiple predictions, significantly
    improving performance (loading takes 2-5 seconds).

    Attributes:
        _instance (Optional[LLMModelCache]): Singleton instance.
        _models (dict): Cache of loaded models and tokenizers.
    """

    _instance: Optional["LLMModelCache"] = None

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def get_model(
        self, model_name: str, quantization: Optional[str] = "4bit"
    ) -> Tuple[Any, Any]:
        """Get or load a model and tokenizer.

        Args:
            model_name (str): HuggingFace model identifier.
            quantization (Optional[str]): Quantization strategy ('4bit', '8bit', or None).

        Returns:
            Tuple: (model, tokenizer)

        Raises:
            ImportError: If transformers or torch are not installed.
            RuntimeError: If model loading fails.
        """
        cache_key = f"{model_name}_{quantization}"

        if cache_key in self._models:
            return self._models[cache_key]

        # Import here to avoid hard dependency
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            import torch
        except ImportError as e:
            raise ImportError(
                "LLM text explanation requires 'transformers' and 'torch'. "
                "Install them with: poetry install --with llm"
            ) from e

        try:
            # Configure quantization
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif quantization == "8bit":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Cache the model and tokenizer
            self._models[cache_key] = (model, tokenizer)

            return model, tokenizer

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {model_name}: {str(e)}"
            ) from e

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._models.clear()
