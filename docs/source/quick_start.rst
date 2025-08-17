Quick Start
===========

This guide will get you started with MLExplainer in just a few steps.

Basic Usage
-----------

Here's a simple example using the binary classification explainer:

Import Dependencies
~~~~~~~~~~~~~~~~~~~

First, import all necessary libraries for the explainer and model training.

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from mlexplainer.explainers.shap.binary import BinaryMLExplainer

Load and Prepare Data
~~~~~~~~~~~~~~~~~~~~~

Load your dataset and separate features from the target variable.

.. code-block:: python

   # Load your data
   df = pd.read_csv('your_data.csv')
   X = df.drop('target', axis=1)
   y = df['target']

Split Dataset
~~~~~~~~~~~~~

Split your data into training and testing sets for proper model validation.

.. code-block:: python

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train Model
~~~~~~~~~~~

Train your machine learning model on the training data.

.. code-block:: python

   # Train a Random Forest model
   model = RandomForestClassifier(random_state=42)
   model.fit(X_train, y_train)

Create Explainer
~~~~~~~~~~~~~~~~

Initialize the MLExplainer with your training data, features, and trained model.

.. code-block:: python

   # Initialize the binary classification explainer
   explainer = BinaryMLExplainer(
       x_train=X_train,
       y_train=y_train,
       features=list(X_train.columns),
       model=model
   )

Generate Explanations
~~~~~~~~~~~~~~~~~~~~~

Generate SHAP-based explanations using quantile analysis.

.. code-block:: python

   # Generate SHAP-based explanations with 5 quantiles
   explanations = explainer.explain(q=5)

Key Concepts
------------

**Explainer Classes**
   Choose the appropriate explainer for your task:
   
   * ``BinaryMLExplainer`` for binary classification
   * ``MultilabelMLExplainer`` for multilabel classification

**Feature Types**
   MLExplainer automatically categorizes your features into:
   
   * Numerical features (continuous values)
   * Categorical features (discrete categories) 
   * String features (text-based, treated as categorical)

**Explanation Types**
   
   * **Global**: Overall feature importance across all predictions
   * **Local**: Individual feature contributions for specific instances

Next Steps
----------

* Learn more about different explainers in the :doc:`explainers` section
* See detailed examples in the :doc:`explainers_examples` section