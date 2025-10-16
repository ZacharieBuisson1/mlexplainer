# Predictors Architecture - Inference Mode

## Overview

This document explains the architecture and design decisions for the **Predictors** module, which enables single-observation inference with SHAP contributions. This is distinct from the **Explainers** module, which focuses on global model analysis and visualization.

## Key Distinction: Explainers vs Predictors

| Aspect | Explainers | Predictors |
|--------|-----------|-----------|
| **Use Case** | Global model analysis & debugging | Production inference |
| **Input** | Full dataset (x_train, y_train) | Single observation (dict/JSON) |
| **Output** | Visualizations & insights | JSON-serializable dict |
| **Features** | Explicitly provided | Auto-detected from model |
| **Pipeline** | Not supported | Optional preprocessing |
| **Focus** | Understanding model behavior | Explaining predictions |

## Architecture

### Base Class: `BaseMLPredictor`

Located in `mlexplainer/core/base_predictor.py`, this abstract base class provides:

1. **Auto-detection of features** from model metadata (XGBoost, LightGBM, CatBoost, scikit-learn)
2. **Auto-detection of categorical features** from model-specific attributes
3. **Raw data preprocessing** via optional scikit-learn Pipeline
4. **Type conversion** for categorical features

### Implementation Classes

#### `BinaryMLPredictor`

Located in `mlexplainer/predictors/binary_predictor.py`

**Output format:**
```json
{
    "prediction": 0.78,
    "contributions": {
        "age": 0.15,
        "income": 0.21,
        "education": -0.03
    }
}
```

#### `MultilabelMLPredictor`

Located in `mlexplainer/predictors/multilabel_predictor.py`

**Output format:**
```json
{
    "label_A": {
        "prediction": 0.78,
        "contributions": {"age": 0.15, "income": 0.21}
    },
    "label_B": {
        "prediction": 0.42,
        "contributions": {"age": -0.05, "income": 0.10}
    }
}
```

## SHAP Values Handling

### Key Insight

SHAP values from `TreeExplainer` have specific structures:

- **Binary classification**:
  - List format: `[negative_class_matrix, positive_class_matrix]`
  - Matrix format: `(n_observations, n_features)`
  - For single observation: Take `[0]` or `[-1][0]` (positive class)

- **Multiclass classification**:
  - List format: `[class_0_matrix, class_1_matrix, ..., class_n_matrix]`
  - Each matrix: `(n_observations, n_features)`
  - Array format: `(n_observations, n_features, n_classes)`
  - For single observation: Extract per-class via `[i][0]` or `[0, :, i]`

### Implementation Strategy

Instead of creating a custom `calculate_single()` method in `ShapWrapper`, we:

1. Use the standard `calculate()` method
2. Handle indexing in the predictor classes
3. Extract scalar values from potentially nested arrays

This approach:
- ✅ Reduces code duplication
- ✅ Maintains compatibility with SHAP library behavior
- ✅ Simplifies debugging and maintenance

## Example Usage

### Binary Classification

```python
from mlexplainer import BinaryMLPredictor
import xgboost as xgb

# Train model
model = xgb.XGBClassifier()
model.fit(x_train, y_train)

# Create predictor (features auto-detected)
predictor = BinaryMLPredictor(model, x_train)

# Predict on raw data
observation = {
    'age': 35,
    'income': 50000,
    'education': 'Bachelor'
}

result = predictor.predict_with_contributions(observation)
# Returns: {"prediction": 0.78, "contributions": {...}}
```

### With Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create preprocessing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler())
])
pipeline.fit(x_train_raw)

# Predictor handles preprocessing automatically
predictor = BinaryMLPredictor(model, x_train_processed, pipeline=pipeline)

# Can now send raw, unprocessed data
raw_observation = {'age': '35', 'income': '50000'}  # Strings OK
result = predictor.predict_with_contributions(raw_observation)
```

### Multiclass Classification

```python
from mlexplainer import MultilabelMLPredictor

# Create predictor with custom label names
predictor = MultilabelMLPredictor(
    model,
    x_train,
    label_names=['Class_A', 'Class_B', 'Class_C']
)

result = predictor.predict_with_contributions(observation)
# Returns: {"Class_A": {...}, "Class_B": {...}, "Class_C": {...}}
```

## Design Decisions

### Why Not Include `base_value`?

Initially, the output included SHAP's `expected_value` (base value). However:
- It adds complexity without significant value for production inference
- Users primarily need: **prediction** and **feature contributions**
- Base value is internal to SHAP's calculation and less interpretable

### Why Not Include `contributions_pct` and `contributions_cumsum`?

These were removed to keep the API simple and focused:
- **Percentages** can be calculated client-side if needed
- **Cumulative sum** is mainly for validation, not production use
- Simpler output = easier integration with downstream systems

### Why Auto-detect Features?

Manual feature specification is error-prone in production. Auto-detection:
- Ensures consistency between training and inference
- Reduces configuration errors
- Supports multiple ML frameworks seamlessly

## Testing Strategy

Tests cover:
- ✅ XGBoost (binary and multiclass)
- ✅ LightGBM (binary and multiclass, with categorical features)
- ✅ CatBoost (binary and multiclass, with categorical features)
- ✅ scikit-learn RandomForest (binary)
- ✅ scikit-learn GradientBoosting (binary)
- ✅ Raw data input (dict, JSON, DataFrame)
- ✅ Preprocessing pipelines
- ✅ Categorical feature handling across frameworks

## Future Enhancements

Potential additions:
- Support for regression tasks
- Batch prediction mode (multiple observations)
- Async prediction support
- Custom SHAP explainer types (e.g., KernelExplainer for non-tree models)
- Confidence intervals for contributions

---

**Author**: Zacharie Buisson
**Date**: October 2024
**Version**: 1.1.0+
