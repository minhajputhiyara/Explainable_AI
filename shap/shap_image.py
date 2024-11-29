import shap
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet50
from matplotlib import pyplot as plt

# Load pre-trained model
model = resnet50.ResNet50(weights="imagenet")

# Load and preprocess an image
img_path = './cat.jpg'  # Replace with your image path
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = resnet50.preprocess_input(img_array)

# Create SHAP explainer for images
explainer = shap.GradientExplainer((model, model.input), img_array)

# Local explanation (single image)
shap_values = explainer.shap_values(img_array)
shap.image_plot(shap_values, np.expand_dims(img, axis=0))  # Local explanation visualization
