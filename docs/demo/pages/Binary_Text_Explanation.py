"""Binary Classification Text Explanation Demo page for MLExplainer application."""

import json
import os

from numpy import nan
from pandas import read_csv
import streamlit as st
from xgboost import XGBClassifier

from mlexplainer.predictors import BinaryMLPredictor


st.set_page_config(
    page_title="Binary Text Explanation - MLExplainer Demo", page_icon="💬"
)


def load_dataset(dataset_name: str):
    """Load selected dataset for binary classification."""
    if dataset_name == "titanic":
        if "titanic_text_dataset" not in st.session_state:
            dataset = read_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "datasets",
                    "binary_datasets",
                    "titanic.csv",
                ),
                sep=",",
            )
            target = "Survived"

            st.session_state["titanic_text_dataset"] = dataset.copy()
            st.session_state["titanic_text_target"] = target

        st.markdown(
            "You chose the well known Titanic dataset. The link for the "
            "dataset is here: "
            "[Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)"
        )

    elif dataset_name == "bank":
        if "bank_text_dataset" not in st.session_state:
            dataset = read_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "datasets",
                    "binary_datasets",
                    "bank.csv",
                ),
                sep=",",
            )
            target = "Exited"

            st.session_state["bank_text_dataset"] = dataset.copy()
            st.session_state["bank_text_target"] = target

        st.markdown(
            "You chose the Bank Marketing dataset. The main goal is to predict banking churn. "
            "The link for the dataset is here: "
            "[Bank Marketing Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)"
        )

    return None


def main():
    st.title("💬 Binary Classification - Text Explanation Demo")
    st.info(
        (
            "This demo showcases the new text explanation feature of MLExplainer. "
            "Make predictions on individual observations and get natural language explanations "
            "powered by SHAP values and template-based or LLM-based generation."
        ),
        icon="✨",
    )
    st.markdown(
        """
    This demo demonstrates the **BinaryMLPredictor** with the `predict_with_text_explanation()` method.

    **Key Features:**
    - 📊 Make predictions on single observations
    - 🔍 Get SHAP contribution values for each feature
    - 💬 Generate natural language explanations in French or English
    - ⚡ Choose between template mode (fast) or LLM mode (intelligent)
    """
    )

    # Dataset selection
    st.header("1. Dataset Selection")
    dataset_options = ["titanic", "bank"]
    selected_dataset = st.selectbox(
        "Choose a dataset for binary classification:",
        dataset_options,
        help="Select a dataset to train a model and generate text explanations",
    )

    if selected_dataset:
        load_dataset(selected_dataset)

        df = st.session_state[f"{selected_dataset}_text_dataset"]
        target = st.session_state[f"{selected_dataset}_text_target"]
        feature_names = [col for col in df.columns.tolist() if col != target]

        st.code(
            (
                "# Load the given dataset\n"
                "from pandas import read_csv\n"
                f"{selected_dataset}_dataset = read_csv(PATH, sep=',')"
            ),
            language="python",
        )

        st.success(f"Loaded {selected_dataset}")
        st.write("First 5 rows:")
        st.dataframe(df.head(5))

        dataset = df[feature_names]
        dataset_target = df[target]
        st.session_state["text_dataset"] = dataset.copy()
        st.session_state["text_dataset_target"] = dataset_target.copy()

        # Model Training
        st.markdown("---")
        st.header("2. Model Training")

        st.session_state["text_selected_features"] = [
            feature
            for feature in feature_names
            if feature
            not in [
                target,
                "PassengerId",
                "Name",
                "Ticket",
                "RowNumber",
                "CustomerId",
                "Surname",
            ]
        ]
        st.write("Selected features for training:")
        st.write(st.session_state["text_selected_features"])

        st.code(
            (
                "# Train XGBoost model\n"
                "from xgboost import XGBClassifier\n"
                "model = XGBClassifier(n_estimators=100, random_state=42)\n"
                "model.fit(x_train, y_train)"
            ),
            language="python",
        )

        if st.button("Train XGBoost Model", key="train_btn"):
            train = dataset.copy()
            train = train.fillna(nan)
            with st.spinner("Training model..."):
                for feature in st.session_state["text_selected_features"]:
                    if train[feature].dtype == "object":
                        train[feature] = train[feature].astype("category")
                    elif str(train[feature].dtype).lower() == "int64":
                        train[feature] = train[feature].astype(int)

                st.session_state["text_train"] = train

                model = XGBClassifier(
                    n_estimators=100, random_state=42, enable_categorical=True
                )
                model.fit(
                    train[st.session_state["text_selected_features"]],
                    dataset_target,
                )

                st.session_state["text_model"] = model
                st.success("✅ Model trained successfully!")

        # Prediction with Text Explanation
        if "text_model" in st.session_state:
            st.markdown("---")
            st.header("3. Single Observation Prediction with Text Explanation")

            st.markdown(
                """
            Now let's make a prediction on a single observation and generate a natural language explanation.

            **How it works:**
            1. Initialize the BinaryMLPredictor with your trained model
            2. Provide a single observation (as dict or DataFrame row)
            3. Choose explanation mode: **template** (fast) or **llm** (intelligent)
            4. Get structured JSON output + natural language explanation
            """
            )

            st.code(
                (
                    "# Initialize predictor\n"
                    "from mlexplainer.predictors import BinaryMLPredictor\n"
                    "predictor = BinaryMLPredictor(model=model, x_train=x_train)\n\n"
                    "# Make prediction with text explanation\n"
                    "observation = {'feature1': value1, 'feature2': value2, ...}\n"
                    "result = predictor.predict_with_text_explanation(\n"
                    "    observation=observation,\n"
                    "    mode='template',  # or 'llm'\n"
                    "    language='fr',    # or 'en'\n"
                    "    top_n=3,\n"
                    "    target_name='churn'\n"
                    ")"
                ),
                language="python",
            )

            # Create input form for observation
            st.subheader("📝 Configure Your Observation")

            col1, col2 = st.columns(2)
            with col1:
                mode = st.radio(
                    "Explanation Mode:",
                    ["template", "llm"],
                    help="Template: Fast, deterministic. LLM: Intelligent reformulation (requires model download)",
                )
                language = st.radio(
                    "Language:",
                    ["fr", "en"],
                    help="Choose French or English explanations",
                )

            with col2:
                top_n = st.slider(
                    "Number of top features to explain:",
                    min_value=1,
                    max_value=min(5, len(st.session_state["text_selected_features"])),
                    value=3,
                )
                target_name = st.text_input(
                    "Target name (for explanation):",
                    value="churn" if selected_dataset == "bank" else "survival",
                )

            # Feature name mapping (optional)
            with st.expander("🔧 Advanced: Feature Name Mapping (Optional)"):
                st.markdown(
                    "Provide human-readable names for technical feature names. "
                    "This helps generate more natural explanations."
                )
                feature_mapping = {}

                if selected_dataset == "bank":
                    st.write("Example mapping for bank dataset:")
                    if st.checkbox("Use default bank mapping"):
                        feature_mapping = {
                            "NumOfProducts": "nombre de produits",
                            "IsActiveMember": "statut de membre actif",
                            "Age": "âge du client",
                            "Balance": "solde du compte",
                        }
                        st.json(feature_mapping)

                if not feature_mapping:
                    mapping_json = st.text_area(
                        "Enter custom mapping as JSON (optional):",
                        value="{}",
                        help='Example: {"NumOfProducts": "nombre de produits", "Age": "âge"}',
                    )
                    try:
                        feature_mapping = json.loads(mapping_json)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
                        feature_mapping = {}

            # Select or create observation
            st.subheader("🎯 Select an Observation")

            observation_source = st.radio(
                "Observation source:",
                ["Sample from dataset", "Custom observation"],
            )

            if observation_source == "Sample from dataset":
                row_index = st.number_input(
                    "Select row index from dataset:",
                    min_value=0,
                    max_value=len(st.session_state["text_train"]) - 1,
                    value=0,
                )
                observation = st.session_state["text_train"][
                    st.session_state["text_selected_features"]
                ].iloc[row_index].to_dict()
                st.write("Selected observation:")
                st.json(observation)

            else:
                st.write("Create custom observation:")
                observation = {}
                cols = st.columns(2)
                for idx, feature in enumerate(st.session_state["text_selected_features"]):
                    with cols[idx % 2]:
                        feature_type = st.session_state["text_train"][feature].dtype
                        if feature_type == "object" or str(feature_type) == "category":
                            unique_values = st.session_state["text_train"][
                                feature
                            ].unique()
                            observation[feature] = st.selectbox(
                                f"{feature}:", options=unique_values, key=f"custom_{feature}"
                            )
                        else:
                            min_val = float(
                                st.session_state["text_train"][feature].min()
                            )
                            max_val = float(
                                st.session_state["text_train"][feature].max()
                            )
                            mean_val = float(
                                st.session_state["text_train"][feature].mean()
                            )
                            observation[feature] = st.number_input(
                                f"{feature}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"custom_{feature}",
                            )

            # Generate prediction and explanation
            if st.button("🚀 Generate Prediction & Explanation", type="primary"):
                predictor = BinaryMLPredictor(
                    model=st.session_state["text_model"],
                    x_train=st.session_state["text_train"][
                        st.session_state["text_selected_features"]
                    ],
                )

                with st.spinner(
                    f"Generating prediction and explanation (mode: {mode})..."
                ):
                    try:
                        result = predictor.predict_with_text_explanation(
                            observation=observation,
                            mode=mode,
                            language=language,
                            top_n=top_n,
                            target_name=target_name,
                            feature_name_mapping=feature_mapping,
                        )

                        # Display results
                        st.success("✅ Prediction complete!")

                        # Main result
                        st.markdown("---")
                        st.subheader("📊 Prediction Result")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Predicted Probability",
                                f"{result['prediction']:.2%}",
                                help="Probability of positive class (1)",
                            )
                        with col2:
                            prediction_label = (
                                "Yes" if result["prediction"] >= 0.5 else "No"
                            )
                            st.metric("Prediction", prediction_label)

                        # Text explanation (highlighted)
                        st.markdown("---")
                        st.subheader("💬 Natural Language Explanation")
                        st.info(result["explanation_text"], icon="💡")

                        # Detailed breakdown
                        with st.expander("🔍 Detailed SHAP Contributions"):
                            st.markdown("**Top Contributing Features:**")
                            for idx, feature_info in enumerate(
                                result["top_features"], 1
                            ):
                                impact_emoji = (
                                    "📈" if feature_info["impact"] == "positive" else "📉"
                                )
                                st.markdown(
                                    f"{idx}. {impact_emoji} **{feature_info['name']}** "
                                    f"(value: `{feature_info['value']}`) → "
                                    f"Contribution: `{feature_info['contribution']:+.4f}`"
                                )

                            st.markdown("**All SHAP Contributions:**")
                            contributions_df = st.session_state[
                                "text_train"
                            ].iloc[:1].copy()
                            contributions_df.loc[0] = list(
                                result["contributions"].values()
                            )
                            st.dataframe(contributions_df)

                        # Full JSON output
                        with st.expander("📄 Full JSON Output"):
                            st.json(result)

                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        if mode == "llm":
                            st.warning(
                                "⚠️ LLM mode requires downloading a 3GB model. "
                                "Try 'template' mode for instant results without downloads."
                            )


if __name__ == "__main__":
    main()
