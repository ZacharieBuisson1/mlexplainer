# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses Poetry for dependency management and packaging.

### Setup and Installation
```bash
poetry install                    # Install dependencies
poetry shell                     # Activate virtual environment
```

### Code Quality Tools
```bash
poetry run black .              # Format code
poetry run isort .               # Sort imports
poetry run pylint mlexplainer   # Lint code
poetry run mypy mlexplainer      # Type checking
```

### Testing
```bash
python -m unittest discover tests/    # Run all tests
python -m unittest tests.test_shap_explainer.TestShapExplainer.test_get_index_valid_column  # Run specific test
```

### Build and Distribution
```bash
poetry build                     # Build package
poetry publish                  # Publish to PyPI (requires auth)
```

## Architecture Overview

This is a machine learning explainability library focused on SHAP (SHapley Additive exPlanations) values for model interpretation.

### Core Architecture

**Base Classes:**
- `mlexplainer.shap_explainer.base.BaseMLExplainer` - Abstract base class for all explainers
- `mlexplainer.shap_explainer.base.ShapWrapper` - Wrapper for SHAP value calculations

**Explainer Implementations:**
- `mlexplainer.shap_explainer.binary_explainers.BinaryMLExplainer` - Binary classification explainer
- Support for multilabel classification in `multilabels_classification/`

**Key Components:**
- **Base Module** (`base/`): Core abstractions and SHAP wrapper functionality
- **Binary Classification** (`binary_classification/`): Validation utilities for binary tasks
- **Multilabel Classification** (`multilabels_classification/`): Feature interpretation for multilabel tasks
- **Plots Module** (`plots/`): Visualization utilities for SHAP values and feature-target relationships

### Data Flow

1. **Initialization**: Explainer classes accept training data (`x_train`, `y_train`), features list, and model
2. **Feature Processing**: Automatic categorization into numerical, categorical, and string features
3. **SHAP Calculation**: Uses `ShapWrapper` to compute SHAP values for specified features
4. **Explanation Generation**: 
   - Global feature importance (mean absolute SHAP values)
   - Local explanations for numerical and categorical features
5. **Visualization**: Integrated plotting for feature-target relationships and SHAP value distributions

### Feature Types Support

The library automatically handles:
- **Numerical features**: Continuous variables with quantile-based analysis
- **Categorical features**: Discrete categories with group-based analysis  
- **String features**: Text-based features treated as categorical

### Key Design Patterns

- **Template Method**: `BaseMLExplainer.explain()` defines the workflow, subclasses implement specific steps
- **Strategy Pattern**: Different plotting strategies for numerical vs categorical features
- **Validation**: Built-in interpretation consistency validation for binary classification

### Testing Strategy

Tests are organized in `tests/` with comprehensive coverage of:
- Utility functions (`test_shap_explainer.py`)
- Edge cases (NaN handling, empty data)
- Feature interpretation validation
- Quantile calculations and grouping logic