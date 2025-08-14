"""Welcome page for MLExplainer Demo application."""

import streamlit as st

st.set_page_config(page_title="Welcome - MLExplainer Demo", page_icon="🏠")


def main():
    st.set_page_config(layout="wide")

    st.title("🏠 Welcome to MLExplainer")

    st.markdown(
        """
    ## About MLExplainer
    
    MLExplainer is a Python library focused on machine learning explainability using 
    **SHAP (SHapley Additive exPlanations)** values for model interpretation.
    
    ### Key Features
    
    - **Binary Classification Explanations**: Detailed SHAP-based explanations for binary classification models
    - **Multilabel Classification Support**: Advanced explanations for multilabel classification tasks
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
    
    ### Getting Started
    
    Use the sidebar navigation to explore:
    - **Binary Classification**: Interactive demo with sample binary classification data
    - **Multilabel Classification**: Interactive demo with sample multilabel data
    
    Each demo page allows you to:
    - Load and explore sample datasets
    - Configure model parameters
    - Generate and visualize explanations
    - Download results and visualizations
    """
    )

    st.info(
        """
    💡 **Tip**: Start with the Binary Classification demo to get familiar with the basic concepts, 
    then explore the Multilabel Classification demo for more advanced use cases.
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
