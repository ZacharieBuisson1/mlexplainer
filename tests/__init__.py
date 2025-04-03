import sys
import os

# Add the src directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)

sys.path.append("../src")
sys.path.append("../src/ml_explainer/shap_explainer")
sys.path.append("../src/ml_explainer/shap_explainer/utils_shap_explainer")
