# Explainable AI: Techniques and Implementations

This repository provides a collection of simple and easy-to-understand implementations for popular **Explainable AI (XAI)** techniques, designed to enhance model interpretability and provide insights into black-box machine learning models. The implementations include both classic and modern methods, covering a range of techniques for various types of model explainability, from feature importance to visual interpretation.

## Key Features

- **LIME (Local Interpretable Model-agnostic Explanations)**:  
  Generate locally faithful explanations for any machine learning model by approximating it with an interpretable surrogate model.
  
- **SHAP (SHapley Additive exPlanations)**:  
  Use Shapley values to explain the output of a machine learning model by assigning each feature a contribution value based on cooperative game theory.
  
- **Counterfactual Explanations**:  
  Provide intuitive explanations by showing what changes to the input would have led to a different model output.
  
- **Occlusion**:  
  Visualize the impact of different parts of the input (e.g., pixels in an image) on the model's predictions by systematically occluding portions of the input.
  
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**:  
  Visualize which regions of an image contributed most to the decision made by convolutional neural networks (CNNs), making it easier to interpret deep learning models.
  
- **Adversarial Attacks**:  
  Demonstrate the vulnerability of machine learning models to adversarial perturbations, helping to understand the limits of model robustness.
  
- **FGSM (Fast Gradient Sign Method)**:  
  Implement a simple method for generating adversarial examples by using gradients from the model's loss function.
  
- **Random Forest Interpretability**:  
  Provide insights into how Random Forest models make decisions through feature importance metrics and decision-path visualizations.
