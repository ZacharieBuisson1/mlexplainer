"""Multilabel Classification Demo page for MLExplainer application."""

import os

from numpy import nan
from pandas import DataFrame, read_csv
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from xgboost import XGBClassifier

from mlexplainer.explainers.shap.multilabel import MultilabelMLExplainer

st.set_page_config(
    page_title="Multilabel Classification - MLExplainer Demo", page_icon="📈"
)


def load_dataset(dataset_name: str):
    """Load selected dataset for multilabel classification."""
    if dataset_name == "iris":
        if "iris_dataset" not in st.session_state:
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

            # cache the dataset and target in Streamlit's session state
            st.session_state["iris_dataset"] = dataset.copy()
            st.session_state["iris_target"] = target

        st.markdown(
            "You chose the well known Iris dataset. The link for the"
            "dataset is here : "
            "[Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)"
        )

    return None


def main():
    st.title("Multilabel Classification Demo")
    st.info(
        (
            "This demo is still work in progress. If you find any issue, open in github "
            "issue [here](https://github.com/ZacharieBuisson1/mlexplainer/issues) in the repo or send me an email : zacharie.buisson@orange.fr"
        ),
        icon="ℹ️",
    )
    st.markdown(
        """
        This demo showcases the MLExplainer library for multilabel classification tasks.
        Select a dataset, train a model, and explore SHAP-based explanations.
        """
    )

    # Dataset selection dropdown
    st.header("1. Dataset Selection")
    dataset_options = ["iris"]
    selected_dataset = st.selectbox(
        "Choose a dataset for multilabel classification:",
        dataset_options,
        help="Each dataset has different characteristics to demonstrate various explanation scenarios",
    )

    if selected_dataset:

        # Load the selected dataset in memory
        load_dataset(selected_dataset)

        # Retrieve dataset and target from session state
        df = st.session_state[f"{selected_dataset}_dataset"]
        target = st.session_state[f"{selected_dataset}_target"]
        feature_names = [col for col in df.columns.tolist() if col != target]

        st.code(
            (
                "# Load the given dataset \n"
                "from pandas import read_csv \n"
                f"{selected_dataset}_dataset = read_csv(PATH, sep=',')"
            ),
            language="python",
        )

        st.success(f"Loaded {selected_dataset}")
        st.write("First 2 rows:")
        st.dataframe(df.head(2))

        # Prepare features for explanation
        dataset = df[feature_names]

        # Encode target labels
        label_encoder = LabelEncoder()
        dataset_target = label_encoder.fit_transform(df[target])
        # Ensure dataset_target is a pandas DataFrame
        dataset_target = DataFrame(dataset_target, columns=[target]).squeeze()
        st.session_state["label_encoder"] = label_encoder

        st.session_state["dataset"] = dataset.copy()
        st.session_state["dataset_target"] = dataset_target.copy()

        # Train-test split
        st.markdown("---")
        st.header("2. Model Training")

        st.session_state["selected_features"] = [
            feature
            for feature in feature_names
            if feature
            not in [
                target,
                "Id",
            ]
        ]
        st.write("List of preselected features :")
        st.write(st.session_state["selected_features"])

        st.code(
            (
                "# Model trained using XGBoostClassifier with 50 estimators and random_state=42. \n"
                "model = XGBClassifier(n_estimators=50, random_state=42) \n"
                "model.fit(dataset, dataset_target)\n"
            ),
            language="python",
        )

        if st.button("Train XGBoost Model"):
            train = dataset.copy()
            train = train.fillna(nan)
            with st.spinner("Training model..."):
                for feature in st.session_state["selected_features"]:
                    if train[feature].dtype == "object":
                        train[feature] = train[feature].astype("category")

                    elif str(train[feature].dtype).lower() == "int64":
                        train[feature] = train[feature].astype(int)

                st.session_state["train"] = train

                # Train XGBoost
                model = XGBClassifier(
                    n_estimators=50, random_state=42, enable_categorical=True
                )
                model.fit(
                    train[st.session_state["selected_features"]],
                    dataset_target,
                )

                st.session_state["model"] = model
                st.success("Model trained successfully!")

        if "model" in st.session_state:
            st.markdown("---")
            st.header("3. Features Interpretation with Shapley's values.")
            st.markdown("How to use the package ?")

            st.code(
                (
                    (
                        """multilabel_explainer = MultilabelMLExplainer(model=fitted_model, x_train=X_train[selected_features],"""
                        """ y_train=y_train, features=selected_features, global_explainer=True, local_explainer=True) \n"""
                        """multilabel_explainer.explain(features_to_explain=["PetalLengthCm"], q=5, figsize=(40, 20))"""
                    )
                ),
                language="python",
            )

            st.write(
                f"Let's interprate one feature of the {selected_dataset} dataset (PetalLengthCm)"
            )

            if selected_dataset == "iris":
                st.session_state["selected_feature_for_example"] = [
                    "PetalLengthCm"
                ]

            multilabel_explainer = MultilabelMLExplainer(
                model=st.session_state["model"],
                x_train=st.session_state["train"][
                    st.session_state["selected_features"]
                ],
                y_train=st.session_state["dataset_target"],
                features=st.session_state["selected_features"],
                global_explainer=False,
                local_explainer=True,
            )
            multilabel_explainer.explain(
                features_to_explain=st.session_state[
                    "selected_feature_for_example"
                ],
                q=5,
                figsize=(45, 25),
                demo_mode=True,
            )
            st.markdown(
                (
                    "This is the final graph of the interpretation of the PetalLengthCm feature (continuous feature) in the iris dataset. \n\n"
                    " To interprate every target modalities, we decided to do one 'one VS all' encoding for the target : 0 vs 1 & 2, 1 vs 0 & 2 and 2 VS 0 & 1. "
                    "Then, every sub graph represents one interpretation, that is to say the impact of every modality in the prediction of each class. \n\n"
                    "How to read this graph, at least every single subgraph :\n"
                    " - The graph shows the impact of the PetalLengthCm feature on the model's predictions.\n"
                    " - The x-axis represents the different values of the PetalLengthCm feature.\n"
                    " - The left y-axis (purple graph, axis and points) represents the predicted probabilities for each value.\n"
                    " - The right y-axis (blue and orange graph, axis and points) represents the SHAP values, indicating the contribution of each value to the prediction.\n\n"
                    " :violet[Observed data of the first sub graph] : Reading purple infos (i.e. the reality), when the size of the petal is below 4, then it is more likely to be class 0, compared to 1 or 2. When greater than 4, it is more likely to be class 1 or 2 (indistinguishable). \n\n"
                    " :orange[SHAP's] :blue[values data of the first sub graph] : Reading orange and blue information, the shap values also indicates that having a 4 or less size increase the probability of being predicted in the class 0 by the model. Having a 4 or more size decrease the probability of being predicted in the class 0 and increase the probability of being predicted in class 1 or 2.\n\n"
                    " Conclusion : both observed data and understanding of the data by the model (i.e. with Shapley's values) is the same for the specific graph 0 VS 1 & 2. The feature PetalLengthCm is well understood by the model to predict the class 0.\n"
                )
            )

            st.markdown("---")
            st.markdown("Play with the feature interpretation : ")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state["local_mode"] = st.checkbox(
                    "Local Mode - Explain individual predictions",
                    value=st.session_state.get("local_mode", True),
                    key="local_checkbox",
                )
            with col2:
                st.session_state["global_mode"] = st.checkbox(
                    "Global Mode - Explain model behavior",
                    value=st.session_state.get("global_mode", True),
                    key="global_checkbox",
                )

            if st.session_state["local_mode"]:

                st.session_state["features_to_explain"] = st.multiselect(
                    "Select features to explain:",
                    options=st.session_state["selected_features"],
                    default=st.session_state["selected_features"],
                    key="features_to_explain_multiselect",
                )
            q = int(st.slider("Select a quantile for SHAP values:", 5, 10, 20))

            if st.button("Explain XGBoost Model"):
                multilabel_explainer = MultilabelMLExplainer(
                    model=st.session_state["model"],
                    x_train=st.session_state["train"][
                        st.session_state["selected_features"]
                    ],
                    y_train=st.session_state["dataset_target"],
                    features=st.session_state["selected_features"],
                    global_explainer=st.session_state["global_mode"],
                    local_explainer=st.session_state["local_mode"],
                )

                with st.spinner("Explanation with SHAP values..."):
                    multilabel_explainer.explain(
                        q=q,
                        features_to_explain=st.session_state[
                            "features_to_explain"
                        ],
                        demo_mode=True,
                        figsize=(40, 20),
                    )


if __name__ == "__main__":
    main()
