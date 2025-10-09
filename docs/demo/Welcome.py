"""Welcome page for MLExplainer Demo application."""

import streamlit as st

st.set_page_config(page_title="Welcome - MLExplainer Demo", page_icon="🏠")


def main():
    st.set_page_config(layout="wide")

    st.title("🏠 Welcome to MLExplainer")
    st.info(
        (
            "This demo is still work in progress. If you find any issue, open in github "
            "issue [here](https://github.com/ZacharieBuisson1/mlexplainer/issues) in the repo or send me an email : zacharie.buisson@orange.fr"
        ),
        icon="ℹ️",
    )
    st.markdown(
        """
    ## About MLExplainer
    
    MLExplainer is a Python library focused on machine learning explainability using 
    **SHAP (SHapley Additive exPlanations)** values for model interpretation.
    
    ### Key Features

    - **Binary Classification Explanations**: Detailed SHAP-based explanations for binary classification models
    - **Multilabel Classification Support**: Advanced explanations for multilabel classification tasks
    - **Text Explanation Generation** ✨ NEW: Natural language explanations powered by SHAP + LLM/Templates
    - **Single Observation Predictions**: BinaryMLPredictor & MultilabelMLPredictor for production inference
    - **Feature Type Handling**: Automatic processing of numerical, categorical, and string features
    - **Visualization Tools**: Rich plotting capabilities for feature-target relationships and SHAP values
    - **Validation Framework**: Built-in interpretation consistency validation
    
    ### Architecture Overview
    
    The library is built around core abstractions:
    
    - `BaseMLExplainer`: Abstract base class for all explainers
    - `ShapWrapper`: Wrapper for SHAP value calculations
    - `BinaryMLExplainer`: Specialized explainer for binary classification
    - `MultilabelMLExplainer`: Specialized explainer for multilabel classification
    
    ### What You Can Do in This Demo

    This demonstration application showcases the key capabilities of MLExplainer:

    **📊 Classic Explainers (Global Model Analysis)**

    1. **Binary Classification Demo**:
       - Load sample data for binary classification
       - Train a model and generate SHAP explanations
       - Visualize global feature importance
       - Explore numerical and categorical feature relationships
       - Validate interpretation consistency

    2. **Multilabel Classification Demo**:
       - Work with multilabel classification datasets
       - Generate SHAP explanations for multiple classes
       - Visualize class-specific feature importance
       - Analyze feature behavior across different labels

    **💬 Text Explanations (Single Observation Inference)** ✨ NEW

    3. **Binary Text Explanation Demo**:
       - Make predictions on individual observations
       - Get SHAP contribution values for each feature
       - Generate natural language explanations (French/English)
       - Choose between Template mode (fast) or LLM mode (intelligent)
       - Understand which features drove a specific prediction

    4. **Multilabel Text Explanation Demo**:
       - Predict probabilities across multiple classes for one observation
       - Get per-label SHAP contributions and text explanations
       - Compare feature importance across different labels
       - Ideal for production inference with explanations
    
    ### Getting Started

    Use the sidebar navigation to explore:

    **Classic Explainers (Global Analysis):**
    - **Binary Classification**: Interactive demo with sample binary classification data
    - **Multilabel Classification**: Interactive demo with sample multilabel data

    **Text Explanations (Inference):** ✨ NEW
    - **Binary Text Explanation**: Make predictions with natural language explanations
    - **Multilabel Text Explanation**: Multi-class predictions with per-label explanations

    Each demo page allows you to:
    - Load and explore sample datasets
    - Train models (XGBoost)
    - Generate SHAP-based explanations
    - (Text demos) Get natural language explanations for individual predictions
    - Visualize results interactively
    """
    )

    st.info(
        """
    💡 **Tips**:
    - **New users**: Start with Binary Classification to understand SHAP basics
    - **Production use cases**: Check out Binary/Multilabel Text Explanation demos for inference
    - **Advanced users**: Explore Multilabel Classification for complex scenarios
    """
    )

    st.markdown(
        """
    ### Technical Details
    
    - **SHAP Integration**: Leverages the SHAP library for generating explanations
    - **Flexible Model Support**: Works with any scikit-learn compatible model
    - **Data Processing**: Automatic feature type detection and preprocessing
    - **Visualization**: Rich matplotlib-based plotting with customizable options
    - **Validation**: Built-in correctness checks for interpretation quality
    """
    )


if __name__ == "__main__":
    main()
