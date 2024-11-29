from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# Load dataset and train model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier().fit(X, y)

# Create a LIME explainer
explainer = LimeTabularExplainer(X, feature_names=data.feature_names, class_names=data.target_names)

# Explain a prediction
instance = X[0]  # Pick a single instance
explanation = explainer.explain_instance(instance, model.predict_proba, num_features=2)

# Visualize the explanation
explanation.show_in_notebook()  # For Jupyter Notebook
# explanation.show_in_notebook(show_table=True)  # Add feature table
