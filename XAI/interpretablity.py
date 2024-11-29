import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load Dataset
data = load_iris()
X, y = data.data, data.target

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X, y)

# Extract feature importance (coefficients)
feature_importance = model.coef_

# Display feature importance for each class
print("Feature Importance (coefficients) for each class:")
for idx, class_coef in enumerate(feature_importance):
    print(f"Class {idx}:")
    for feature, coef in zip(data.feature_names, class_coef):
        print(f"  {feature}: {coef:.4f}")

# Local explanation for a single prediction
instance = X[0]  # Select a single data point
probabilities = model.predict_proba([instance])
predicted_class = np.argmax(probabilities)
print(f"\nLocal Explanation for instance {instance}:")
print(f"Predicted Class: {data.target_names[predicted_class]}")
print("Feature Contributions (Feature Value * Coefficient):")
for feature, value, coef in zip(data.feature_names, instance, feature_importance[predicted_class]):
    contribution = value * coef
    print(f"  {feature}: {contribution:.4f}")
