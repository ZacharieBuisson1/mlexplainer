"""LLM-based text explanation generator using Qwen2.5-1.5B-Instruct."""

from typing import Any, Dict, Optional

from mlexplainer.core.base_text_explainer import BaseTextExplainer
from mlexplainer.interpretation.model_cache import LLMModelCache


class TextExplainerLLM(BaseTextExplainer):
    """LLM-based text explanation generator.

    Uses Qwen2.5-1.5B-Instruct to generate fluent, natural language explanations
    of model predictions with feature contributions.

    Features:
    - Intelligent variable name reformulation (NumOfProducts → "nombre de produits")
    - Concise explanations (2-4 lines)
    - Context-aware interpretation of feature impacts
    - Multi-language support (French and English)

    Attributes:
        model_name (str): HuggingFace model identifier.
        quantization (str): Quantization strategy ('4bit', '8bit', or None).
        max_new_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature for generation.
        model_cache (LLMModelCache): Singleton cache for model instances.
    """

    # System prompts per language
    SYSTEM_PROMPTS = {
        "fr": """Tu es un expert en analyse prédictive. Ta mission : expliquer les prédictions d'un modèle de machine learning de manière claire et professionnelle.

Règles strictes :
- Commence TOUJOURS par "La probabilité de [cible] est de X%."
- Mentionne EXACTEMENT les 3 facteurs listés ci-dessous
- Reformule les noms techniques en français courant SEULEMENT si c'est évident (ex: NumOfProducts → nombre de produits)
- Si le nom est ambigu ou déjà clair, garde-le TEL QUEL (ex: Geography_France → Geography_France)
- Indique clairement l'impact de chaque facteur (augmente/réduit)
- Maximum 3 phrases au total
- Ton professionnel mais accessible""",
        "en": """You are an expert in predictive analysis. Your mission: explain machine learning model predictions clearly and professionally.

Strict rules:
- ALWAYS start with "The probability of [target] is X%."
- Mention EXACTLY the 3 factors listed below
- Reformulate technical names into plain English ONLY if obvious (e.g., NumOfProducts → number of products)
- If name is ambiguous or already clear, keep it AS IS (e.g., Geography_France → Geography_France)
- Clearly indicate each factor's impact (increases/decreases)
- Maximum 3 sentences total
- Professional but accessible tone""",
    }

    # User prompt templates per language
    USER_PROMPT_TEMPLATES = {
        "fr": """Probabilité prédite : {prediction:.0%} pour "{target_name}"

Facteurs explicatifs (par ordre d'importance) :
{features_section}

Génère l'explication en suivant CE FORMAT EXACT :

"La probabilité de [reformuler target_name] est de {prediction:.0%}. [Expliquer facteur 1 avec reformulation + impact]. [Expliquer facteurs 2 et 3 avec reformulations + impacts]."

Exemples de reformulations (REFORMULE UNIQUEMENT SI ÉVIDENT) :
- NumOfProducts → le nombre de produits souscrits (REFORMULER)
- IsActiveMember → le statut de membre actif (REFORMULER)
- Age → l'âge du client (REFORMULER)
- Geography_France → Geography_France (GARDER TEL QUEL - pas évident)
- feature_x123 → feature_x123 (GARDER TEL QUEL - cryptique)""",
        "en": """Predicted probability: {prediction:.0%} for "{target_name}"

Explanatory factors (by importance):
{features_section}

Generate the explanation following THIS EXACT FORMAT:

"The probability of [reformulate target_name] is {prediction:.0%}. [Explain factor 1 with reformulation + impact]. [Explain factors 2 and 3 with reformulations + impacts]."

Reformulation examples (REFORMULATE ONLY IF OBVIOUS):
- NumOfProducts → the number of subscribed products (REFORMULATE)
- IsActiveMember → active member status (REFORMULATE)
- Age → customer age (REFORMULATE)
- Geography_France → Geography_France (KEEP AS IS - not obvious)
- feature_x123 → feature_x123 (KEEP AS IS - cryptic)""",
    }

    def __init__(
        self,
        language: str = "fr",
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        quantization: Optional[str] = None,  # Default None for Mac compatibility
        max_new_tokens: int = 300,
        temperature: float = 0.5,  # Balanced between coherence and creativity
    ):
        """Initialize the LLM-based text explainer.

        Args:
            language (str): Language for explanations ('fr' or 'en'). Default: 'fr'.
            model_name (str): HuggingFace model identifier. Default: 'Qwen/Qwen2.5-1.5B-Instruct'.
            quantization (str): Quantization strategy ('4bit', '8bit', or None). Default: '4bit'.
            max_new_tokens (int): Maximum tokens to generate. Default: 300.
            temperature (float): Sampling temperature (0.0-1.0). Default: 0.3.

        Raises:
            ImportError: If transformers or torch are not installed.
            ValueError: If language is invalid.
        """
        super().__init__(language)

        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Get cached model and tokenizer (lazy-loaded)
        self.model_cache = LLMModelCache()

    def generate_explanation(
        self,
        prediction: float,
        contributions: Dict[str, float],
        values: Dict[str, Any],
        top_n: int = 3,
        target_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a natural language explanation using LLM.

        Args:
            prediction (float): The model's prediction (probability).
            contributions (Dict[str, float]): Feature SHAP contributions.
            values (Dict[str, Any]): Feature values (after preprocessing).
            top_n (int): Number of top contributing features to include. Default: 3.
            target_name (Optional[str]): Human-readable name for the target variable.
            **kwargs (Any): Additional parameters:
                - max_new_tokens (int): Override default max tokens.
                - temperature (float): Override default temperature.

        Returns:
            str: Natural language explanation of the prediction.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If LLM generation fails.
        """
        # Set default target name
        if target_name is None:
            target_name = (
                "positive class" if self.language == "en" else "classe positive"
            )

        # Extract top features
        top_features = self._extract_top_features(contributions, values, top_n)

        # Build features section for prompt
        features_section = self._format_features_section(top_features)

        # Build prompts
        system_prompt = self.SYSTEM_PROMPTS[self.language]
        user_prompt = self.USER_PROMPT_TEMPLATES[self.language].format(
            prediction=prediction,
            target_name=target_name,
            top_n=top_n,
            features_section=features_section,
        )

        # Get model and tokenizer
        model, tokenizer = self.model_cache.get_model(
            self.model_name, self.quantization
        )

        # Generate explanation
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Override generation parameters if provided
            max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
            temp = kwargs.get("temperature", self.temperature)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode only the NEW tokens (excluding the input prompt)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            explanation = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Clean up any remaining chat template artifacts
            explanation = explanation.strip()

            # Remove common artifacts at the start
            if explanation.startswith("assistant"):
                explanation = explanation[len("assistant"):].strip()
            if explanation.startswith("\n"):
                explanation = explanation.strip()

            # Remove incomplete sentences at the start (if starts with lowercase or special char)
            while explanation and (explanation[0].islower() or explanation[0] in '"),.\'"'):
                # Find first sentence start (capital letter after period or start)
                first_capital = -1
                for i, char in enumerate(explanation):
                    if char.isupper() and (i == 0 or explanation[i-1] in '.!?'):
                        first_capital = i
                        break

                if first_capital > 0:
                    explanation = explanation[first_capital:].strip()
                else:
                    break  # No cleanup needed

            return explanation

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

    def _format_features_section(self, top_features: list) -> str:
        """Format the features section for the LLM prompt.

        Args:
            top_features (list): List of feature dictionaries.

        Returns:
            str: Formatted features section.
        """
        lines = []
        for i, feature in enumerate(top_features, 1):
            name = feature["name"]
            value = feature["value"]
            contrib = feature["contribution"]
            impact = feature["impact"]

            # Format contribution as percentage
            contrib_pct = abs(contrib) * 100
            sign = "+" if impact == "positive" else "-"

            line = f"{i}. {name} = {value} → contribution: {sign}{contrib_pct:.1f}%"
            lines.append(line)

        return "\n".join(lines)
