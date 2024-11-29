import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier().fit(X, y)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Global explanation (feature importance)
shap_values = explainer.shap_values(X)  # Compute SHAP values for the dataset
shap.summary_plot(shap_values, X, feature_names=data.feature_names)  # Global summary plot

# Local explanation (single prediction)
instance = X[0]  # Single instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=data.feature_names)
