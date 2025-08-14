"""Binary Classification Demo page for MLExplainer application."""

import os
import sys

from numpy import nan
from pandas import read_csv
import streamlit as st
from xgboost import XGBClassifier

from mlexplainer.explainers.shap.binary import BinaryMLExplainer


st.set_page_config(
    page_title="Binary Classification - MLExplainer Demo", page_icon="📊"
)


def load_dataset(dataset_name: str):
    """Load selected dataset for binary classification."""
    if dataset_name == "titanic":
        if "titanic_dataset" not in st.session_state:
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

            # cache the dataset and target in Streamlit's session state
            st.session_state["titanic_dataset"] = dataset.copy()
            st.session_state["titanic_target"] = target

        st.markdown(
            "You chose the well known Titanic dataset. The link for the"
            "dataset is here : "
            "[Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)"
        )

    elif dataset_name == "bank":
        if "bank_dataset" not in st.session_state:
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

            # cache the dataset and target in Streamlit's session state
            st.session_state["bank_dataset"] = dataset.copy()
            st.session_state["bank_target"] = target

        st.markdown(
            "You chose the Bank Marketing dataset. The main goal is to predict banking churn."
            "The link for the dataset is here : "
            "[Bank Marketing Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)"
        )

    return None


def main():
    st.title("Binary Classification Demo")
    st.markdown(
        """
    This demo showcases the MLExplainer library for binary classification tasks.
    Select a dataset, train a model, and explore SHAP-based explanations.
    """
    )

    # Dataset selection dropdown
    st.header("1. Dataset Selection")
    dataset_options = ["titanic", "bank"]
    selected_dataset = st.selectbox(
        "Choose a dataset for binary classification:",
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
        dataset_target = df[target]
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
                "PassengerId",
                "Name",
                "Ticket",
                "RowNumber",
                "CustomerId",
                "Surname",
            ]
        ]
        st.write("List of preselected features :")
        st.write(st.session_state["selected_features"])

        st.code(
            (
                "# Model trained using XGBoostClassifier with 100 estimators and random_state=42. \n"
                "model = XGBClassifier(n_estimators=100, random_state=42) \n"
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
                    n_estimators=100, random_state=42, enable_categorical=True
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
            st.markdown("How to use the package ? :")

            st.code(
                (
                    (
                        """binary_explainer = BinaryMLExplainer(model=model, x_train=dataframe[selected_features], y_train=dataframe_target, """
                        """features=selected_features, global_explainer=True, local_explainer=True) \n"""
                        f"""binary_explainer.explain(features_to_explain=["{'Sex' if selected_dataset == "titanic" else "Age"}"])"""
                    )
                ),
                language="python",
            )

            st.write(
                f"Let's interprate one feature of the {selected_dataset} dataset"
            )

            if selected_dataset == "titanic":
                st.session_state["selected_feature_for_example"] = ["Sex"]
            if selected_dataset == "bank":
                st.session_state["selected_feature_for_example"] = ["Age"]

            binary_explainer = BinaryMLExplainer(
                model=st.session_state["model"],
                x_train=st.session_state["train"][
                    st.session_state["selected_features"]
                ],
                y_train=st.session_state["dataset_target"],
                features=st.session_state["selected_features"],
                global_explainer=False,
                local_explainer=True,
            )
            binary_explainer.explain(
                features_to_explain=st.session_state[
                    "selected_feature_for_example"
                ],
                demo_mode=True,
            )
            if selected_dataset == "titanic":
                st.markdown(
                    (
                        "This is the final graph of the interpretation of the sex feature (discrete feature) in the titanic dataset. "
                        "How to read this graphe :\n"
                        " - The graph shows the impact of the sex feature on the model's predictions.\n"
                        " - The x-axis represents the different categories of the sex feature.\n"
                        " - The left y-axis (purple graph, axis and points) represents the predicted probabilities for each category.\n"
                        " - The right y-axis (blue and orange graph, axis and points) represents the SHAP values, indicating the contribution of each category to the prediction.\n\n"
                        " :violet[Observed data] : Reading purple infos (i.e. the reality), the female category has an average predicted probability (violet point) above the global mean of the target (hyphen purple line). The male category is below the mean. \n\n"
                        " :orange[SHAP's] :blue[values data] : Reading orange and blue information, the shap values also indicates that being a woman has a positive impact on the prediction of survival in average, while being a man has a negative impact. \n\n"
                        " Conclusion : both observed data and understanding of the data by the model (i.e. with Shapley's values) is the same. The feature Sex is well understood.\n"
                    )
                )
            if selected_dataset == "bank":
                st.markdown(
                    (
                        "This is the final graph of the interpretation of the age feature (continuous feature) in the bank dataset. "
                        "How to read this graphe :\n"
                        " - The graph shows the impact of the sex feature on the model's predictions.\n"
                        " - The x-axis represents the different categories of the sex feature.\n"
                        " - The left y-axis (purple graph, axis and points) represents the predicted probabilities for each category.\n"
                        " - The right y-axis (blue and orange graph, axis and points) represents the SHAP values, indicating the contribution of each category to the prediction.\n\n"
                        " :violet[Observed data] : Reading purple infos (i.e. the reality),when the age is lower than 40, the target mean for churn is below the global mean of churn. When the age is higher than 40, the churn rate is higher than the global mean. \n\n"
                        " :orange[SHAP's] :blue[values data] : Reading orange and blue information, the shap values also indicates that being less than 40 y.o has a negative impact on the prediction of churn in average, while being more than 40 y.o has a positive impact. We can also see very few observation after 70 y.o that are less churner than the mean. \n\n"
                        " Conclusion : Both observed data and understanding of the data by the model (i.e. with Shapley's values) is the same. The feature Age is well understood. We can also say that the model has a good understanding of extreme values.\n"
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
                binary_explainer = BinaryMLExplainer(
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
                    binary_explainer.explain(
                        q=q,
                        features_to_explain=st.session_state[
                            "features_to_explain"
                        ],
                        demo_mode=True,
                    )


if __name__ == "__main__":
    main()
