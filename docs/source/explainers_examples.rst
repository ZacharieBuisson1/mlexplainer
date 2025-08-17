Explainer Examples
==================

This section provides practical examples of using different explainers in MLExplainer.

Binary Classification Example
------------------------------

Complete Binary Classification Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a comprehensive example using the Binary SHAP explainer with a real dataset:

Data Preparation
^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_classification
   from mlexplainer.explainers.shap.binary import BinaryMLExplainer

   # Create a sample dataset
   X, y = make_classification(
       n_samples=1000,
       n_features=10,
       n_informative=7,
       n_redundant=3,
       n_classes=2,
       random_state=42
   )
   
   # Convert to DataFrame with feature names
   feature_names = [f'feature_{i}' for i in range(X.shape[1])]
   df = pd.DataFrame(X, columns=feature_names)
   df['target'] = y

Model Training
^^^^^^^^^^^^^^

.. code-block:: python

   # Prepare features and target
   X = df.drop('target', axis=1)
   y = df['target']
   
   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42, stratify=y
   )
   
   # Train Random Forest model
   model = RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       random_state=42
   )
   model.fit(X_train, y_train)

Explanation Generation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize the explainer
   explainer = BinaryMLExplainer(
       x_train=X_train,
       y_train=y_train,
       features=list(X_train.columns),
       model=model
   )
   
   # Generate explanations with quantile analysis
   explanations = explainer.explain(q=5)

Accessing Results
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Access feature importance (global explanations)
   feature_importance = explanations['feature_importance']
   print("Global Feature Importance:")
   for feature, importance in feature_importance.items():
       print(f"{feature}: {importance:.4f}")
   
   # Access numerical features analysis
   numerical_features = explanations['numerical_features']
   print("\nNumerical Features Analysis:")
   for feature, analysis in numerical_features.items():
       print(f"{feature}: {len(analysis)} quantile groups")
   
   # Access categorical features (if any)
   categorical_features = explanations['categorical_features']
   print(f"\nCategorical Features: {len(categorical_features)}")

Multilabel Classification Example
----------------------------------

Multilabel Classification Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working with multilabel classification tasks:

Data Setup
^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import make_multilabel_classification
   from sklearn.multioutput import MultiOutputClassifier
   from mlexplainer.explainers.shap.multilabel import MultilabelMLExplainer

   # Create multilabel dataset
   X, y = make_multilabel_classification(
       n_samples=800,
       n_features=12,
       n_classes=3,
       n_labels=2,
       random_state=42
   )
   
   # Convert to DataFrame
   feature_names = [f'feature_{i}' for i in range(X.shape[1])]
   X_df = pd.DataFrame(X, columns=feature_names)

Model Training
^^^^^^^^^^^^^^

.. code-block:: python

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X_df, y, test_size=0.3, random_state=42
   )
   
   # Train multilabel model
   base_model = RandomForestClassifier(n_estimators=50, random_state=42)
   multilabel_model = MultiOutputClassifier(base_model)
   multilabel_model.fit(X_train, y_train)

Explanation Generation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize multilabel explainer
   explainer = MultilabelMLExplainer(
       x_train=X_train,
       y_train=y_train,
       features=list(X_train.columns),
       model=multilabel_model
   )
   
   # Generate explanations
   explanations = explainer.explain(q=4)

Advanced Usage
--------------

Custom Feature Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Working with mixed feature types:

.. code-block:: python

   # Create dataset with mixed feature types
   data = {
       'numerical_1': np.random.normal(0, 1, 500),
       'numerical_2': np.random.exponential(2, 500),
       'categorical_1': np.random.choice(['A', 'B', 'C'], 500),
       'categorical_2': np.random.choice(['X', 'Y'], 500),
       'string_feature': np.random.choice(['type1', 'type2', 'type3'], 500)
   }
   
   df_mixed = pd.DataFrame(data)
   df_mixed['target'] = (
       (df_mixed['numerical_1'] > 0) & 
       (df_mixed['categorical_1'] == 'A')
   ).astype(int)

Model and Explainer Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Prepare features (encode categorical variables)
   from sklearn.preprocessing import LabelEncoder
   
   df_encoded = df_mixed.copy()
   label_encoders = {}
   
   for col in ['categorical_1', 'categorical_2', 'string_feature']:
       le = LabelEncoder()
       df_encoded[col] = le.fit_transform(df_mixed[col])
       label_encoders[col] = le
   
   X_mixed = df_encoded.drop('target', axis=1)
   y_mixed = df_encoded['target']
   
   # Train model and create explainer
   X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(
       X_mixed, y_mixed, test_size=0.3, random_state=42
   )
   
   model_mixed = RandomForestClassifier(random_state=42)
   model_mixed.fit(X_train_mixed, y_train_mixed)
   
   explainer_mixed = BinaryMLExplainer(
       x_train=X_train_mixed,
       y_train=y_train_mixed,
       features=list(X_train_mixed.columns),
       model=model_mixed
   )

Detailed Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Generate detailed explanations
   explanations_mixed = explainer_mixed.explain(q=6)
   
   # Analyze feature types automatically detected
   print("Detected Feature Types:")
   print(f"Numerical: {len(explanations_mixed['numerical_features'])}")
   print(f"Categorical: {len(explanations_mixed['categorical_features'])}")
   
   # Access quantile-based analysis for numerical features
   for feature, analysis in explanations_mixed['numerical_features'].items():
       print(f"\n{feature} quantile analysis:")
       for quantile_info in analysis:
           print(f"  Range: {quantile_info['range']}")
           print(f"  Count: {quantile_info['count']}")
           print(f"  Mean SHAP: {quantile_info['mean_shap']:.4f}")

Tips and Best Practices
------------------------

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

* Use appropriate ``q`` values (3-10) for quantile analysis
* Consider sampling large datasets before explanation generation
* Cache explainer objects for repeated analysis

Feature Selection
~~~~~~~~~~~~~~~~~

* Remove highly correlated features before explanation
* Consider feature importance for dimensionality reduction
* Validate explanations consistency for binary classification

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

* Focus on features with high absolute SHAP values
* Compare local vs global explanations
* Validate explanations against domain knowledge