"""Multilabel Classification Text Explanation Demo page for MLExplainer application."""

import json
import os

from numpy import nan
from pandas import DataFrame, read_csv
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from xgboost import XGBClassifier

from mlexplainer.predictors import MultilabelMLPredictor


st.set_page_config(
    page_title="Multilabel Text Explanation - MLExplainer Demo", page_icon="💬"
)


def load_dataset(dataset_name: str):
    """Load selected dataset for multilabel classification."""
    if dataset_name == "iris":
        if "iris_text_dataset" not in st.session_state:
            dataset = read_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "datasets",
                    "multilabel_datasets",
                    "iris.csv",
                ),
                sep=",",
            )
            target = "Species"

            st.session_state["iris_text_dataset"] = dataset.copy()
            st.session_state["iris_text_target"] = target

        st.markdown(
            "You chose the well known Iris dataset. The link for the "
            "dataset is here: "
            "[Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)"
        )

    return None


def main():
    st.title("💬 Multilabel Classification - Text Explanation Demo")
    st.info(
        (
            "This demo showcases the new text explanation feature of MLExplainer for multilabel tasks. "
            "Make predictions on individual observations and get natural language explanations "
            "for **each label** powered by SHAP values."
        ),
        icon="✨",
    )
    st.markdown(
        """
    This demo demonstrates the **MultilabelMLPredictor** with the `predict_with_text_explanation()` method.

    **Key Features:**
    - 📊 Make predictions on single observations across multiple labels
    - 🔍 Get SHAP contribution values for each feature **per label**
    - 💬 Generate natural language explanations in French or English
    - ⚡ Choose between template mode (fast) or LLM mode (intelligent)
    - 🎯 Understand how features contribute differently to each class
    """
    )

    # Dataset selection
    st.header("1. Dataset Selection")
    dataset_options = ["iris"]
    selected_dataset = st.selectbox(
        "Choose a dataset for multilabel classification:",
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

        # Encode target labels
        label_encoder = LabelEncoder()
        dataset_target = label_encoder.fit_transform(df[target])
        dataset_target = DataFrame(dataset_target, columns=[target]).squeeze()
        st.session_state["label_encoder"] = label_encoder
        st.session_state["label_names"] = label_encoder.classes_.tolist()

        st.session_state["text_ml_dataset"] = dataset.copy()
        st.session_state["text_ml_dataset_target"] = dataset_target.copy()

        # Model Training
        st.markdown("---")
        st.header("2. Model Training")

        st.session_state["text_ml_selected_features"] = [
            feature for feature in feature_names if feature not in [target, "Id"]
        ]
        st.write("Selected features for training:")
        st.write(st.session_state["text_ml_selected_features"])
        st.write(f"Labels: {st.session_state['label_names']}")

        st.code(
            (
                "# Train XGBoost multiclass model\n"
                "from xgboost import XGBClassifier\n"
                "model = XGBClassifier(n_estimators=50, random_state=42)\n"
                "model.fit(x_train, y_train)"
            ),
            language="python",
        )

        if st.button("Train XGBoost Model", key="train_ml_btn"):
            train = dataset.copy()
            train = train.fillna(nan)
            with st.spinner("Training model..."):
                for feature in st.session_state["text_ml_selected_features"]:
                    if train[feature].dtype == "object":
                        train[feature] = train[feature].astype("category")
                    elif str(train[feature].dtype).lower() == "int64":
                        train[feature] = train[feature].astype(int)

                st.session_state["text_ml_train"] = train

                model = XGBClassifier(
                    n_estimators=50, random_state=42, enable_categorical=True
                )
                model.fit(
                    train[st.session_state["text_ml_selected_features"]],
                    dataset_target,
                )

                st.session_state["text_ml_model"] = model
                st.success("✅ Model trained successfully!")

        # Prediction with Text Explanation
        if "text_ml_model" in st.session_state:
            st.markdown("---")
            st.header("3. Single Observation Prediction with Text Explanation")

            st.markdown(
                """
            Now let's make a prediction on a single observation and generate natural language explanations
            **for each label**.

            **How it works:**
            1. Initialize the MultilabelMLPredictor with your trained model
            2. Provide a single observation (as dict or DataFrame row)
            3. Choose explanation mode: **template** (fast) or **llm** (intelligent)
            4. Get structured JSON output + natural language explanation **per label**
            """
            )

            st.code(
                (
                    "# Initialize predictor\n"
                    "from mlexplainer.predictors import MultilabelMLPredictor\n"
                    "predictor = MultilabelMLPredictor(\n"
                    "    model=model,\n"
                    "    x_train=x_train,\n"
                    "    label_names=['Setosa', 'Versicolor', 'Virginica']\n"
                    ")\n\n"
                    "# Make prediction with text explanation\n"
                    "observation = {'feature1': value1, 'feature2': value2, ...}\n"
                    "result = predictor.predict_with_text_explanation(\n"
                    "    observation=observation,\n"
                    "    mode='template',  # or 'llm'\n"
                    "    language='fr',    # or 'en'\n"
                    "    top_n=2\n"
                    ")"
                ),
                language="python",
            )

            # Configuration
            st.subheader("📝 Configure Your Observation")

            col1, col2 = st.columns(2)
            with col1:
                mode = st.radio(
                    "Explanation Mode:",
                    ["template", "llm"],
                    help="Template: Fast, deterministic. LLM: Intelligent reformulation (requires model download)",
                    key="ml_mode",
                )
                language = st.radio(
                    "Language:",
                    ["fr", "en"],
                    help="Choose French or English explanations",
                    key="ml_language",
                )

            with col2:
                top_n = st.slider(
                    "Number of top features to explain (per label):",
                    min_value=1,
                    max_value=min(
                        4, len(st.session_state["text_ml_selected_features"])
                    ),
                    value=2,
                    key="ml_top_n",
                )

            # Feature name mapping (optional)
            with st.expander("🔧 Advanced: Feature Name Mapping (Optional)"):
                st.markdown(
                    "Provide human-readable names for technical feature names. "
                    "This helps generate more natural explanations."
                )
                feature_mapping = {}

                if selected_dataset == "iris":
                    st.write("Example mapping for iris dataset:")
                    if st.checkbox("Use default iris mapping", key="ml_mapping"):
                        feature_mapping = {
                            "SepalLengthCm": "longueur des sépales (cm)",
                            "SepalWidthCm": "largeur des sépales (cm)",
                            "PetalLengthCm": "longueur des pétales (cm)",
                            "PetalWidthCm": "largeur des pétales (cm)",
                        }
                        st.json(feature_mapping)

                if not feature_mapping:
                    mapping_json = st.text_area(
                        "Enter custom mapping as JSON (optional):",
                        value="{}",
                        help='Example: {"SepalLengthCm": "longueur des sépales"}',
                        key="ml_mapping_json",
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
                key="ml_obs_source",
            )

            if observation_source == "Sample from dataset":
                row_index = st.number_input(
                    "Select row index from dataset:",
                    min_value=0,
                    max_value=len(st.session_state["text_ml_train"]) - 1,
                    value=0,
                    key="ml_row_index",
                )
                observation = st.session_state["text_ml_train"][
                    st.session_state["text_ml_selected_features"]
                ].iloc[row_index].to_dict()

                actual_label = st.session_state["label_encoder"].inverse_transform(
                    [st.session_state["text_ml_dataset_target"].iloc[row_index]]
                )[0]
                st.write("Selected observation:")
                st.json(observation)
                st.info(f"**Actual label:** {actual_label}", icon="ℹ️")

            else:
                st.write("Create custom observation:")
                observation = {}
                cols = st.columns(2)
                for idx, feature in enumerate(
                    st.session_state["text_ml_selected_features"]
                ):
                    with cols[idx % 2]:
                        feature_type = st.session_state["text_ml_train"][feature].dtype
                        if feature_type == "object" or str(feature_type) == "category":
                            unique_values = st.session_state["text_ml_train"][
                                feature
                            ].unique()
                            observation[feature] = st.selectbox(
                                f"{feature}:",
                                options=unique_values,
                                key=f"ml_custom_{feature}",
                            )
                        else:
                            min_val = float(
                                st.session_state["text_ml_train"][feature].min()
                            )
                            max_val = float(
                                st.session_state["text_ml_train"][feature].max()
                            )
                            mean_val = float(
                                st.session_state["text_ml_train"][feature].mean()
                            )
                            observation[feature] = st.number_input(
                                f"{feature}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"ml_custom_{feature}",
                            )

            # Generate prediction and explanation
            if st.button(
                "🚀 Generate Prediction & Explanation", type="primary", key="ml_predict"
            ):
                predictor = MultilabelMLPredictor(
                    model=st.session_state["text_ml_model"],
                    x_train=st.session_state["text_ml_train"][
                        st.session_state["text_ml_selected_features"]
                    ],
                    label_names=st.session_state["label_names"],
                )

                with st.spinner(
                    f"Generating predictions and explanations (mode: {mode})..."
                ):
                    try:
                        result = predictor.predict_with_text_explanation(
                            observation=observation,
                            mode=mode,
                            language=language,
                            top_n=top_n,
                            feature_name_mapping=feature_mapping,
                        )

                        # Display results
                        st.success("✅ Prediction complete!")

                        # Main result - probabilities for all labels
                        st.markdown("---")
                        st.subheader("📊 Prediction Results (All Labels)")

                        # Find predicted label (highest probability)
                        label_probs = {
                            label: result[label]["prediction"]
                            for label in st.session_state["label_names"]
                        }
                        predicted_label = max(label_probs, key=label_probs.get)

                        cols = st.columns(len(st.session_state["label_names"]))
                        for idx, label in enumerate(st.session_state["label_names"]):
                            with cols[idx]:
                                is_predicted = label == predicted_label
                                st.metric(
                                    label,
                                    f"{result[label]['prediction']:.2%}",
                                    help=f"Probability of {label}",
                                    delta="Predicted" if is_predicted else None,
                                )

                        st.info(f"**Predicted Class:** {predicted_label}", icon="🎯")

                        # Text explanations per label
                        st.markdown("---")
                        st.subheader("💬 Natural Language Explanations (Per Label)")

                        for label in st.session_state["label_names"]:
                            with st.expander(
                                f"📝 {label} - {result[label]['prediction']:.2%}",
                                expanded=(label == predicted_label),
                            ):
                                st.info(result[label]["explanation_text"], icon="💡")

                                st.markdown("**Top Contributing Features:**")
                                for idx, feature_info in enumerate(
                                    result[label]["top_features"], 1
                                ):
                                    impact_emoji = (
                                        "📈"
                                        if feature_info["impact"] == "positive"
                                        else "📉"
                                    )
                                    st.markdown(
                                        f"{idx}. {impact_emoji} **{feature_info['name']}** "
                                        f"(value: `{feature_info['value']}`) → "
                                        f"Contribution: `{feature_info['contribution']:+.4f}`"
                                    )

                        # Comparison view
                        with st.expander("🔍 Cross-Label Feature Importance Comparison"):
                            st.markdown(
                                "Compare how the same feature contributes differently across labels:"
                            )

                            import pandas as pd

                            comparison_data = []
                            for label in st.session_state["label_names"]:
                                for feature, contrib in result[label][
                                    "contributions"
                                ].items():
                                    comparison_data.append(
                                        {
                                            "Label": label,
                                            "Feature": feature,
                                            "Contribution": contrib,
                                        }
                                    )

                            comparison_df = pd.DataFrame(comparison_data)
                            pivot_table = comparison_df.pivot(
                                index="Feature", columns="Label", values="Contribution"
                            )
                            st.dataframe(pivot_table.style.background_gradient(cmap="RdYlGn", axis=1))

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
