# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You are 🎯 PACT Orchestrator, an expert workflow coordinator specializing in guiding software development through the PACT (Prepare, Architect, Code, Test) framework. You are a strategic orchestrator who coordinates development workflows but does not write code or create files yourself - your expertise lies in delegation and phase management.

# CORE CAPABILITIES

You excel at:
- Thinking prior to each output to full consider your strategy
- Breaking down complex development requests into the four PACT phases
- Identifying which specialists to delegate to for each phase
- Maintaining project state and tracking progress
- Synthesizing outputs from each phase into coherent instructions for the next
- Ensuring quality gates are met before phase transitions

# OPERATIONAL FRAMEWORK

## Phase Structure
0. **Folder Creation**: Create a `docs` folder if it doesn't already exist with `/preparation` and `/architecture` subfolders to house all documentation, and create a project specific file to help document everything happening in the project that you will update after every phase.
1. **Prepare Phase**: Instruct the pact-preparer to use batch tool for Research, documentation gathering, requirement analysis, and creating markdown docs for their findings.
2. **Architect Phase**: Instruct the pact-architect to batch read the pact-preparer's documentation, then to think about system design, component planning, and interface definition before creating markdowns documents of its recommendations.
3. **Code Phase**: Instruct the relevant pack coders (backend, frontend, database-engineer) to read the relevant documentation creater by the pact-preparer and pact-architect, and begin coding.
4. **Test Phase**: Instruct the pact-test-engineer to devise of unit, integration, and e2e tests, then to run them and fix any issues that arise.

## Specilist Roster
- technical-research-engineer : For all the research on the internet 
- website-source-analyzer : For understanding the raw source code of website 
- web-scrapper-architect : For developping scrapers 
- code-optimizer : To correct code and optimize it 
- software-architect : To design the package it must be
- test-engineer : To test properly the code
- database-engineer : To develop database through CSV files
- sql-data-analyst : To analyze data using python or SQL


# EXECUTION PROTOCOL

When receiving a development request, you will:

1. **Assess and Plan**
Analyze the request to determine how it maps to the PACT phases. Consider project complexity, dependencies, and required specialists. Then, if not done already, create your project specific markdown file to document progress.

2. **Create Phase Tasks**
For each phase, define:
   - Specific objectives with measurable outcomes
   - Required inputs from previous phases or external sources
   - Expected outputs and deliverables
   - Clear success criteria for phase completion
   - Any dependencies or prerequisites

3. **Delegate Effectively**
When assigning tasks to specialists:
   - Consider doing a batch request if tasks can be done in parallel
   - Provide comprehensive context from previous phases
   - Include all relevant project documentation to read
   - Specify exact deliverables needed
   - Set clear expectations for output format
   - Highlight any constraints or requirements

4. **Track Project State**
Maintain a clear record of:
   - ✅ Completed phases with key outputs
   - 🔄 Currently active phase and assigned specialist
   - ⏳ Pending phases and their dependencies
   - 🚧 Any blockers, risks, or issues identified
   - 📊 Overall project progress percentage

5. **Synthesize and Transition**
Between phases:
   - Review outputs for completeness and quality
   - Extract key information needed for the next phase
   - Identify any gaps or clarifications needed
   - Ensure smooth context transfer to the next specialist

# COMMUNICATION STANDARDS

When interacting with users, you will:

1. **Project Status Updates**: Provide clear, structured updates including:
   - Current phase and progress
   - Recent accomplishments
   - Active tasks and responsible specialists
   - Upcoming milestones
   - Any decisions needed from the user

2. **Phase Summaries**: After each phase completion:
   - Highlight key deliverables produced
   - Summarize important decisions made
   - Note any deviations from original plan
   - Present artifacts for user review

3. **Recommendation Format**: When suggesting next steps:
   - Explain the rationale based on PACT framework
   - Identify which specialist to engage
   - Outline expected outcomes
   - Estimate effort or timeline if possible

# QUALITY ASSURANCE

You will enforce these quality gates:

- **Prepare Phase**: Requirements are clear, documented in a markdown file, and validated
- **Architect Phase**: Design is complete, scalable, and addresses all requirements in a markdown file
- **Code Phase**: Implementation matches design and meets coding standards
- **Test Phase**: All tests pass and quality metrics are satisfied

If any of these fail, send it back to the specific agent that will be best suited to solving the issues with clear instructions about the problem, and recommended solutions to explore.

# CONSTRAINTS AND LIMITATIONS

- You do NOT write code or create files yourself
- You do NOT make technical implementation decisions - defer to user or specialists
- You do NOT proceed without clear phase completion criteria being met

# ADAPTATION GUIDELINES

While maintaining the PACT sequence, you will adapt your approach based on:
- Project size and complexity of request
- Available specialists and resources
- User preferences and constraints
- Technical stack and requirements
- Timeline and urgency

Remember: Your role is to orchestrate, not implement. You ensure the right specialist does the right work at the right time, maintaining quality and coherence throughout the development lifecycle.


---

# PROJECT INFORMATION

## Project Overview

**MLExplainer** is an advanced Machine Learning package dedicated to model interpretability, with a primary focus on leveraging Shapley values (SHAP) for explaining complex predictive models, particularly boosting algorithms for tabular data.

- **Current Version**: 1.0.1
- **License**: MIT
- **Python Version**: ^3.11
- **Package Repository**: [PyPI - mlexplainer](https://pypi.org/project/mlexplainer/)
- **Documentation**: [ReadTheDocs](https://mlexplainer.readthedocs.io/en/latest/index.html)
- **Demo Application**: [Streamlit Demo](https://mlexplainer.streamlit.app/)
- **Source Repository**: [GitHub](https://github.com/ZacharieBuisson1/mlexplainer)

## Key Features

- **SHAP Integration**: Built-in support for SHAP explainers with optimized workflows using TreeExplainer
- **Multiple Classification Types**: Full support for binary and multilabel classification tasks
- **Inference Mode**: Predictors for single-observation inference with SHAP contributions (NEW in v1.1+)
- **Automatic Feature Detection**: Intelligent categorization of numerical, categorical, and string features
- **Raw Data Support**: Handles JSON, untyped CSV, and supports scikit-learn pipelines for preprocessing
- **Rich Visualizations**: Integrated plotting capabilities for feature-target relationships and SHAP value distributions
- **Validation Tools**: Built-in interpretation consistency validation via `correctness_features()` method
- **Modern Architecture**: Clean, extensible design with proper abstractions and design patterns

---

# DEVELOPMENT ENVIRONMENT

## Dependency Management

This project uses **Poetry** for dependency management and packaging. Poetry handles virtual environments, dependency resolution, and package building/publishing.

## Setup and Installation

```bash
# Install all dependencies (including dev and docs groups)
poetry install

# Activate the virtual environment
poetry shell

# Install only production dependencies
poetry install --only main
```

## Development Commands

### Code Quality Tools

```bash
# Format code with Black (code formatter)
poetry run black .

# Sort imports with isort
poetry run isort .

# Lint code with Pylint
poetry run pylint mlexplainer

# Type checking with mypy
poetry run mypy mlexplainer

# Run all code quality checks together
poetry run black . && poetry run isort . && poetry run pylint mlexplainer && poetry run mypy mlexplainer
```

### Testing

```bash
# Run all tests
python -m unittest discover tests/

# Run a specific test file
python -m unittest tests.test_binary_explainer

# Run a specific test class
python -m unittest tests.test_shap_explainer.TestShapExplainer

# Run a specific test method
python -m unittest tests.test_shap_explainer.TestShapExplainer.test_get_index_valid_column

# Run tests with coverage
poetry run coverage run -m unittest discover tests/
poetry run coverage report
poetry run coverage html  # Generate HTML coverage report
```

### Build and Distribution

```bash
# Build the package (creates wheel and tar.gz in dist/)
poetry build

# Publish to PyPI (requires PyPI authentication token)
poetry publish

# Build and publish in one command
poetry publish --build

# Publish to Test PyPI (for testing before production release)
poetry publish -r testpypi
```

### Documentation

```bash
# Run Streamlit demo locally
poetry run streamlit run docs/demo/Welcome.py

# Build Sphinx documentation (if configured)
cd docs && poetry run sphinx-build -b html source build
```

---

# ARCHITECTURE OVERVIEW

## Project Structure

```
mlexplainer/
├── mlexplainer/              # Main package
│   ├── __init__.py
│   ├── core/                 # Core abstractions
│   │   ├── __init__.py
│   │   ├── base_explainer.py  # Base class for explainers
│   │   └── base_predictor.py  # Base class for predictors (NEW)
│   ├── explainers/           # Explainer implementations (global analysis)
│   │   ├── __init__.py
│   │   ├── lime/             # LIME explainers (placeholder)
│   │   │   └── __init__.py
│   │   └── shap/             # SHAP explainers
│   │       ├── __init__.py
│   │       ├── wrapper.py    # SHAP TreeExplainer wrapper
│   │       ├── binary.py     # Binary classification explainer
│   │       └── multilabel.py # Multilabel classification explainer
│   ├── predictors/           # Predictor implementations (single-obs inference) (NEW)
│   │   ├── __init__.py
│   │   ├── binary_predictor.py    # Binary classification predictor
│   │   └── multilabel_predictor.py # Multilabel classification predictor
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   └── quantiles.py
│   ├── validation/           # Validation tools
│   │   ├── __init__.py
│   │   └── feature_interpretation.py
│   └── visualization/        # Plotting utilities
│       ├── __init__.py
│       ├── target_plots.py
│       └── shap_plots.py
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_binary_explainer.py
│   ├── test_binary_predictor.py  # Tests for binary predictor (NEW)
│   ├── test_core.py
│   ├── test_imports.py
│   ├── test_multilabel_explainer.py
│   ├── test_multilabel_predictor.py  # Tests for multilabel predictor (NEW)
│   ├── test_shap_explainer.py
│   ├── test_shap_wrapper.py
│   ├── test_suite.py
│   ├── test_utils.py
│   └── test_validation.py
├── docs/                     # Documentation
│   ├── demo/                 # Streamlit demo app
│   │   ├── Welcome.py
│   │   └── pages/
│   │       ├── Binary_Classification.py
│   │       └── Multilabel_Classification.py
│   └── source/               # Sphinx documentation source
│       └── conf.py
├── pyproject.toml            # Poetry configuration & dependencies
├── README.md                 # Project readme
└── CLAUDE.md                 # This file
```

## Core Architecture

### Base Classes

- **`mlexplainer.core.base_explainer.BaseMLExplainer`**
  - Abstract base class for all explainers (global model interpretation)
  - Implements the Template Method pattern
  - Defines the core workflow: initialization → feature processing → explanation generation
  - Provides abstract methods for subclasses to implement specific behavior
  - Handles automatic feature type detection (numerical, categorical, string)
  - Used for dataset-level analysis and visualization

- **`mlexplainer.core.base_predictor.BaseMLPredictor`** (NEW)
  - Abstract base class for all predictors (single-observation inference)
  - Auto-detects feature names from model (XGBoost, LightGBM, CatBoost, scikit-learn)
  - Auto-detects categorical features from model metadata
  - Supports raw data input (JSON, dict, untyped DataFrame)
  - Handles preprocessing via optional scikit-learn Pipeline
  - Converts categorical features to proper dtype automatically
  - Used for production inference with SHAP explanations

- **`mlexplainer.explainers.shap.wrapper.ShapWrapper`**
  - Wrapper for SHAP value calculations using `shap.TreeExplainer`
  - Optimized for tree-based models (XGBoost, LightGBM, etc.)
  - Handles SHAP value computation and caching
  - Provides interface for accessing SHAP values across features
  - `calculate()`: Batch SHAP calculation for datasets
  - `calculate_single()`: Single-observation SHAP calculation (NEW)

### Explainer Implementations

- **`mlexplainer.explainers.shap.binary.BinaryMLExplainer`**
  - Extends `BaseMLExplainer` for binary classification tasks
  - Computes SHAP values for positive class predictions
  - Provides feature importance rankings
  - Generates explanations for individual features (numerical and categorical)
  - Includes integrated visualization methods

- **`mlexplainer.explainers.shap.multilabel.MultilabelMLExplainer`**
  - Extends `BaseMLExplainer` for multilabel classification tasks
  - Handles multiple target labels simultaneously
  - Computes SHAP values per label
  - Supports label-specific feature importance analysis
  - Provides cross-label feature comparison capabilities

### Predictor Implementations (NEW)

- **`mlexplainer.predictors.binary_predictor.BinaryMLPredictor`**
  - Extends `BaseMLPredictor` for binary classification inference
  - Predicts probability for positive class
  - Calculates SHAP contributions for each feature
  - Returns structured dict with:
    - `prediction`: Predicted probability (0-1)
    - `base_value`: Expected value (baseline probability)
    - `contributions`: Feature-level SHAP contributions
    - `contributions_pct`: Relative contributions as percentages
    - `contributions_cumsum`: Cumulative contribution verification
  - Supports XGBoost, Random Forest, Gradient Boosting, and other tree-based models
  - Usage: `BinaryMLPredictor(model, x_train, pipeline=None)`

- **`mlexplainer.predictors.multilabel_predictor.MultilabelMLPredictor`**
  - Extends `BaseMLPredictor` for multilabel/multiclass inference
  - Predicts probabilities for multiple labels simultaneously
  - Calculates SHAP contributions per label
  - Returns nested dict: `{label_name: {prediction, contributions, ...}}`
  - Supports custom label names or auto-generates (label_0, label_1, ...)
  - Handles both multiclass (mutually exclusive) and multilabel (independent) tasks
  - Usage: `MultilabelMLPredictor(model, x_train, label_names=['A', 'B'], pipeline=None)`

### Key Modules

#### Core Module (`mlexplainer/core/`)
- Contains foundational abstractions and base classes
- Defines the explainer interface and common functionality
- Ensures consistency across all explainer implementations

#### Explainers Module (`mlexplainer/explainers/`)
- **SHAP submodule**: Primary focus, contains SHAP-based explainers
- **LIME submodule**: Placeholder for future LIME integration
- Modular design allows easy addition of new explainer types
- **Focus**: Global model interpretation and visualization

#### Predictors Module (`mlexplainer/predictors/`) (NEW)
- Contains predictor implementations for single-observation inference
- Complements explainers by providing production-ready prediction API
- Auto-detects features and types from trained models
- Supports raw data input (JSON, dict, DataFrame)
- Optional preprocessing pipeline integration
- Returns structured JSON-serializable results
- **Focus**: Production inference with SHAP explanations

#### Utils Module (`mlexplainer/utils/`)
- **`data_processing.py`**: Data transformation and preprocessing utilities
- **`quantiles.py`**: Quantile calculation for numerical feature analysis
- Helper functions for feature type detection and conversion

#### Validation Module (`mlexplainer/validation/`)
- **`feature_interpretation.py`**: Validates interpretation consistency
- Implements `correctness_features()` method for sanity checks
- Ensures SHAP values align with actual feature-target relationships

#### Visualization Module (`mlexplainer/visualization/`)
- **`target_plots.py`**: Plots for feature-target relationships
- **`shap_plots.py`**: SHAP-specific visualizations (beeswarm, waterfall, etc.)
- Integrated with explainer classes for seamless plotting

## Usage Examples

### Explainers vs Predictors

**Explainers** are for **global model analysis** on datasets:
```python
from mlexplainer import BinaryMLExplainer

# Analyze model behavior on entire dataset
explainer = BinaryMLExplainer(
    x_train=x_train,  # Full training dataset
    y_train=y_train,  # Full training labels
    features=['age', 'income', 'education'],
    model=xgb_model
)

# Generate global insights and visualizations
explainer.explain()  # Creates plots and analysis for all features
correctness = explainer.correctness_features()  # Validate interpretations
```

**Predictors** are for **single-observation inference** in production:
```python
from mlexplainer import BinaryMLPredictor

# Make predictions with SHAP explanations
predictor = BinaryMLPredictor(
    model=xgb_model,
    x_train=x_train,  # For SHAP TreeExplainer initialization
    pipeline=preprocessing_pipeline  # Optional: for raw data preprocessing
)

# Predict on single observation (dict, JSON, or DataFrame)
observation = {
    'age': 35,
    'income': 50000,
    'education': 'Bachelor'
}

result = predictor.predict_with_contributions(observation)
# Returns:
# {
#     'prediction': 0.78,  # Probability
#     'base_value': 0.42,  # Model baseline
#     'contributions': {
#         'age': 0.15,
#         'income': 0.21,
#         'education': 0.03
#     },
#     'contributions_pct': {
#         'age': 41.7,  # % of total contribution
#         'income': 58.3,
#         'education': 8.3
#     },
#     'contributions_cumsum': 0.78
# }
```

### Key Differences

| Aspect | Explainers | Predictors |
|--------|-----------|-----------|
| **Use Case** | Model analysis & debugging | Production inference |
| **Input** | Full dataset (x_train, y_train) | Single observation (dict/JSON) |
| **Output** | Visualizations & global insights | Structured dict (JSON-serializable) |
| **Features** | Must be explicitly provided | Auto-detected from model |
| **Pipeline** | Not supported | Optional preprocessing pipeline |
| **Focus** | Understanding model globally | Explaining individual predictions |

## Data Flow

1. **Initialization**
   - User provides training data (`x_train`, `y_train`), feature list, and trained model
   - Explainer validates inputs and initializes SHAP wrapper
   - Feature types are automatically detected and categorized

2. **Feature Processing**
   - Numerical features identified via dtype analysis
   - Categorical features detected (low cardinality or object dtype)
   - String features treated as categorical
   - Feature metadata stored for downstream processing

3. **SHAP Calculation**
   - `ShapWrapper` computes SHAP values using TreeExplainer
   - Values calculated for all features or specified subset
   - Results cached for efficient reuse

4. **Explanation Generation**
   - **Global importance**: Mean absolute SHAP values across all instances
   - **Local explanations**:
     - Numerical features: Quantile-based analysis via `_explain_numerical()`
     - Categorical features: Group-based analysis via `_explain_categorical()`
   - Feature-target relationship analysis
   - Consistency validation against actual data patterns

5. **Visualization**
   - Automated plot generation for feature distributions
   - SHAP value distributions across feature ranges
   - Target variable correlation plots
   - Interactive visualizations in Streamlit demo

## Feature Types Support

### Numerical Features
- **Detection**: Features with numeric dtypes (int, float) and high cardinality
- **Analysis**: Quantile-based binning for distribution analysis
- **Explanation**: SHAP value trends across feature ranges
- **Visualization**: Line plots, scatter plots with trend lines

### Categorical Features
- **Detection**: Features with object dtype or low cardinality (<10 unique values)
- **Analysis**: Group-based statistics per category
- **Explanation**: SHAP value distributions per category
- **Visualization**: Bar plots, box plots per category

### String Features
- **Detection**: Features with string/object dtype
- **Treatment**: Converted to categorical for analysis
- **Analysis**: Same as categorical features
- **Note**: High-cardinality string features may require preprocessing

## Key Design Patterns

### Template Method Pattern
- **Implementation**: `BaseMLExplainer.explain()` defines the workflow
- **Purpose**: Ensures consistent explanation process across explainer types
- **Flexibility**: Subclasses implement specific steps (`_explain_numerical()`, `_explain_categorical()`)

### Strategy Pattern
- **Implementation**: Different plotting strategies for feature types
- **Purpose**: Adapts visualization approach based on feature characteristics
- **Benefit**: Clean separation of plotting logic from explanation logic

### Wrapper Pattern
- **Implementation**: `ShapWrapper` wraps `shap.TreeExplainer`
- **Purpose**: Provides simplified, consistent interface to SHAP library
- **Benefit**: Isolates SHAP-specific code, easier to maintain and test

### Factory Pattern (Implicit)
- **Implementation**: Explainer selection based on task type (binary vs multilabel)
- **Purpose**: Simplifies explainer instantiation for end users
- **Future**: Could be made explicit with factory class if more explainer types added

## Dependencies

### Core Dependencies
- **numpy** (>=1.24.2, <3.0.0): Numerical computing
- **pandas** (^2.2.3): Data manipulation and analysis
- **shap** (^0.47.1): SHAP value computation
- **matplotlib** (^3.10.1): Base plotting library
- **seaborn** (^0.13.2): Statistical data visualization
- **xgboost** (^3.0.4): Tree-based model support (TreeExplainer optimization)
- **pyarrow** (^19.0.1): Efficient data serialization
- **coverage** (^7.8.0): Test coverage measurement

### Code Quality Dependencies (codequality group)
- **black** (^25.1.0): Code formatter
- **isort** (^6.0.1): Import sorter
- **pylint** (^3.3.6): Linter
- **mypy** (^1.16.1): Static type checker
- **pandas-stubs** (^2.3.0.250703): Type stubs for pandas

### Development Dependencies (dev group)
- **ipykernel** (^6.29.5): Jupyter kernel for notebook development

### Documentation Dependencies (docs group)
- **sphinx** (^8.1.3): Documentation generator
- **sphinx-rtd-theme** (^3.0.2): ReadTheDocs theme
- **sphinx-autodoc-typehints** (^2.5.0): Type hints in documentation
- **streamlit** (^1.48.0): Demo application framework

---

# TESTING STRATEGY

## Test Organization

Tests are located in `tests/` directory with comprehensive coverage across all modules:

### Test Files

1. **`test_core.py`**
   - Tests for `BaseMLExplainer` abstract base class
   - Validates feature detection logic
   - Tests initialization and validation

2. **`test_binary_explainer.py`**
   - Tests for `BinaryMLExplainer`
   - Binary classification scenarios
   - Feature importance calculations
   - Explanation generation for binary targets

3. **`test_multilabel_explainer.py`**
   - Tests for `MultilabelMLExplainer`
   - Multilabel classification scenarios
   - Label-specific SHAP value computation
   - Cross-label feature analysis

4. **`test_shap_explainer.py`**
   - Integration tests for SHAP explainers
   - End-to-end explanation workflows
   - Specific test: `test_get_index_valid_column`

5. **`test_shap_wrapper.py`**
   - Tests for `ShapWrapper` class
   - SHAP value computation accuracy
   - TreeExplainer integration

6. **`test_utils.py`**
   - Tests for utility functions
   - Data processing functions
   - Quantile calculations

7. **`test_validation.py`**
   - Tests for validation module
   - Feature interpretation consistency checks
   - `correctness_features()` method validation

8. **`test_imports.py`**
   - Package import tests
   - Ensures all modules are importable
   - Catches circular import issues

9. **`test_suite.py`**
   - Test suite aggregator
   - Runs all tests in sequence

## Test Coverage Areas

### Functional Testing
- Core explainer functionality
- SHAP value computation
- Feature importance ranking
- Explanation generation (numerical and categorical)
- Visualization generation

### Edge Cases
- **NaN handling**: Missing values in features
- **Empty data**: Zero-length datasets
- **Single-value features**: Constant features
- **High-cardinality categoricals**: Many unique categories
- **Type coercion**: Mixed-type features

### Integration Testing
- End-to-end explanation workflows
- Model-explainer integration (XGBoost, etc.)
- Visualization pipeline integration

### Regression Testing
- Ensures new changes don't break existing functionality
- Critical for maintaining package stability

## Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_binary_explainer

# Run specific test with coverage
poetry run coverage run -m unittest tests.test_shap_explainer
poetry run coverage report
```

---

# BEST PRACTICES FOR CONTRIBUTORS

## Code Style
- Follow **PEP 8** style guidelines
- Use **Black** for consistent formatting (line length: 88 chars)
- Sort imports with **isort** (profile: black)
- Add type hints for all function signatures
- Write descriptive docstrings (Google or NumPy style)

## Testing
- Write tests for all new features
- Aim for >80% code coverage
- Include edge case tests
- Use descriptive test method names

## Documentation
- Update README.md for user-facing changes
- Update CLAUDE.md for architecture changes
- Add docstrings to all public APIs
- Include usage examples in docstrings

## Version Control
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Write clear, descriptive commit messages
- Create feature branches for new work
- Submit pull requests for review

## Release Process
1. Update version in `pyproject.toml`
2. Update CHANGELOG (if exists)
3. Run full test suite: `python -m unittest discover tests/`
4. Run code quality checks: `black . && isort . && pylint mlexplainer && mypy mlexplainer`
5. Build package: `poetry build`
6. Test on Test PyPI: `poetry publish -r testpypi`
7. Publish to PyPI: `poetry publish`
8. Tag release in git: `git tag v1.0.1 && git push --tags`

---

# TROUBLESHOOTING

## Common Issues

### Import Errors
- **Issue**: `ModuleNotFoundError: No module named 'mlexplainer'`
- **Solution**: Ensure you're in the Poetry virtual environment (`poetry shell`) and dependencies are installed (`poetry install`)

### SHAP TreeExplainer Errors
- **Issue**: SHAP failing with tree-based models
- **Solution**: Ensure model is compatible with TreeExplainer (XGBoost, LightGBM, scikit-learn tree models)

### Type Checking Failures
- **Issue**: mypy errors about pandas types
- **Solution**: Install pandas-stubs: `poetry install --with codequality`

### Test Failures
- **Issue**: Tests failing locally but passing in CI
- **Solution**: Check Python version (should be ^3.11), verify dependency versions match `poetry.lock`

---

# CONTACT AND SUPPORT

- **Issues**: Report bugs or suggest features on [GitHub Issues](https://github.com/ZacharieBuisson1/mlexplainer/issues)
- **Author**: Zacharie Buisson <zacharie.buisson@orange.fr>
- **Documentation**: [ReadTheDocs](https://mlexplainer.readthedocs.io/en/latest/index.html)
- **Demo**: [Streamlit Demo](https://mlexplainer.streamlit.app/)