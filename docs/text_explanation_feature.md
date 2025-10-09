# Text Explanation Feature - Natural Language Explanations for Predictions

## Overview

MLExplainer now supports **automatic natural language generation** to explain model predictions. This feature transforms technical SHAP contributions into clear, readable explanations in French or English.

## Two Modes Available

### 1. Template Mode (Recommended for Production)

**Fast, deterministic, no heavy dependencies**

```python
from mlexplainer import BinaryMLPredictor

predictor = BinaryMLPredictor(model, x_train, pipeline=preprocessing_pipeline)

result = predictor.predict_with_text_explanation(
    observation={'Age': 42, 'NumOfProducts': 1, 'IsActiveMember': 1},
    mode='template',
    top_n=3,
    language='fr',
    target_name='Customer Churn',
    feature_name_mapping={
        'NumOfProducts': 'nombre de produits',
        'Age': 'âge du client',
        'IsActiveMember': 'statut de membre actif'
    }
)

print(result['explanation_text'])
# Output: "La probabilité de Customer Churn est de 38%. Cette prédiction s'explique
# principalement par nombre de produits (valeur: 1, contribution: +53%), statut de
# membre actif (valeur: 1, contribution: -48%) et âge du client (valeur: 42, contribution: +24%)."
```

**Advantages:**
- ⚡ Instant generation (<50ms)
- 🎯 Fully controllable output
- 📦 No additional dependencies
- ✅ Production-ready

### 2. LLM Mode (Advanced, Requires Optional Dependencies)

**Intelligent reformulation using local Small Language Models**

```python
result = predictor.predict_with_text_explanation(
    observation={'Age': 42, 'NumOfProducts': 1, 'IsActiveMember': 1},
    mode='llm',
    top_n=3,
    language='fr',
    target_name='Customer Churn'
)

print(result['explanation_text'])
# Output: "La probabilité de sortie du client est de 38%. Le nombre de produits
# souscrits augmente la probabilité (+53%), tandis que le statut de membre actif
# la réduit (-48%). L'âge du client contribue également positivement (+24%)."
```

**Advantages:**
- 🧠 Intelligent variable name reformulation
- 📝 Natural, fluent text generation
- 🔒 100% local (no API calls)
- 🌍 Multilingual (French & English)

**Requirements:**
```bash
poetry install --with llm
```

**Note:** First run downloads the model (~1.5GB). Subsequent runs are fast (~2-5s) thanks to caching.

## Architecture

### Core Components

```
mlexplainer/
├── core/
│   └── base_text_explainer.py       # Abstract base class
├── interpretation/                   # New module
│   ├── text_explainer_template.py   # Template-based generation
│   ├── text_explainer_llm.py        # LLM-based generation
│   └── model_cache.py               # Singleton cache for LLM models
└── predictors/
    ├── binary_predictor.py          # predict_with_text_explanation()
    └── multilabel_predictor.py      # predict_with_text_explanation()
```

### Design Patterns

- **Template Method Pattern**: `BaseTextExplainer` defines the explanation interface
- **Singleton Pattern**: `LLMModelCache` ensures models are loaded only once
- **Strategy Pattern**: Swap between template and LLM modes seamlessly

## LLM Configuration

### Recommended Model: Qwen2.5-1.5B-Instruct

Selected after comprehensive benchmarking of 8+ Small Language Models (see `docs/SLM_Research_Report_2025.md`).

**Why Qwen2.5-1.5B?**
- ✅ Best balance: speed (1-3s) + quality
- ✅ Excellent French support (29+ languages)
- ✅ Optimized for structured output (JSON → text)
- ✅ 1.5B parameters (fits in 3GB RAM with quantization)

### Customization

```python
from mlexplainer.interpretation import TextExplainerLLM

custom_explainer = TextExplainerLLM(
    language='en',
    model_name='Qwen/Qwen2.5-1.5B-Instruct',
    quantization=None,  # or '4bit', '8bit' (requires CUDA)
    temperature=0.5,    # 0.0 = deterministic, 1.0 = creative
    max_new_tokens=300
)

explanation = custom_explainer.generate_explanation(
    prediction=0.78,
    contributions={'feature_a': 0.3, 'feature_b': -0.2},
    values={'feature_a': 42, 'feature_b': 100},
    top_n=3,
    target_name='Churn'
)
```

## Prompt Engineering Strategy

The LLM mode uses **highly structured prompts** to ensure:
- Consistent format: Always starts with "La probabilité de X est de Y%"
- Smart reformulation: Only reformulates obvious technical names
- Ambiguous names preserved: `Geography_France` stays as-is if unclear
- Professional tone: Balanced between technical accuracy and accessibility

Example prompt structure:
```
System: You are an expert in predictive analysis...
Rules:
- Always start with "The probability of [target] is X%."
- Mention EXACTLY the 3 factors listed
- Reformulate ONLY if obvious (NumOfProducts → number of products)
- Keep ambiguous names AS IS (feature_x123 → feature_x123)
...
```

## Performance Benchmarks

| Mode | First Run | Subsequent Runs | Memory Usage | Dependencies |
|------|-----------|----------------|--------------|--------------|
| Template | <50ms | <50ms | ~10MB | None |
| LLM (0.5B) | ~30s (model load) | ~2-3s | ~1.5GB | transformers, torch |
| LLM (1.5B) | ~60s (model load) | ~3-5s | ~3GB | transformers, torch |

## Use Cases

### Template Mode
- ✅ Production APIs (low latency required)
- ✅ Batch processing (thousands of predictions)
- ✅ Resource-constrained environments
- ✅ When variable names are already clear

### LLM Mode
- ✅ Interactive dashboards (acceptable 2-5s latency)
- ✅ Reports & presentations (high-quality explanations)
- ✅ Cryptic variable names (needs intelligent reformulation)
- ✅ Multilingual support without manual mapping

## Limitations & Considerations

### LLM Mode on macOS (Apple Silicon)
- ⚠️ Quantization (4-bit/8-bit) requires CUDA (not available on Mac)
- ✅ Solution: Use `quantization=None` for FP16 mode (works but slower)
- 💡 Tip: Use smaller models (0.5B) for faster inference on Mac

### Text Quality
- Template mode: Deterministic, always consistent
- LLM mode: May vary slightly between runs (use lower temperature for consistency)

## Migration Guide

**Existing code** (without text explanation):
```python
result = predictor.predict_with_contributions(observation)
# Returns: {'prediction': 0.78, 'contributions': {...}}
```

**New code** (with text explanation):
```python
result = predictor.predict_with_text_explanation(
    observation,
    mode='template',  # or 'llm'
    top_n=3,
    language='fr'
)
# Returns: {'prediction': 0.78, 'contributions': {...},
#           'explanation_text': "...", 'top_features': [...]}
```

**Backward compatible:** `predict_with_contributions()` still works unchanged.

## Future Enhancements

Potential improvements for future versions:
- [ ] API-based LLM support (OpenAI GPT-4, Claude, Mistral API)
- [ ] Customizable explanation templates
- [ ] Support for regression tasks
- [ ] Multi-language expansion (Spanish, German, etc.)
- [ ] Explanation caching for repeated predictions

---

**For technical details on model selection, see:** `docs/SLM_Research_Report_2025.md`
