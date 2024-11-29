import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50
import matplotlib.pyplot as plt
import cv2

# Load pre-trained model
model = resnet50.ResNet50(weights="imagenet")

# Load and preprocess an image
img_path = './cat.jpg'  # Replace with your image path
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = resnet50.preprocess_input(img_array)

# Get model predictions
preds = model.predict(img_array)
class_idx = np.argmax(preds[0])  # Predicted class index
class_output = model.output[:, class_idx]  # Output for the predicted class

# Get the last convolutional layer
last_conv_layer = model.get_layer("conv5_block3_out")

# Compute gradients of the class output w.r.t. the feature map
grads = tf.GradientTape()  # To compute gradients
with tf.GradientTape() as tape:
    tape.watch(last_conv_layer.output)
    preds = model(img_array)
    class_output = preds[:, class_idx]

# Get gradients
grads = tape.gradient(class_output, last_conv_layer.output)[0]

# Compute the weights for the heatmap
weights = tf.reduce_mean(grads, axis=(0, 1))

# Create the Grad-CAM heatmap
feature_map = last_conv_layer.output[0]
heatmap = tf.reduce_sum(weights * feature_map, axis=-1).numpy()
heatmap = np.maximum(heatmap, 0)  # ReLU
heatmap /= np.max(heatmap)  # Normalize to [0, 1]

# Superimpose heatmap on original image
heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

# Display results
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(superimposed_img[:, :, ::-1])  # Convert BGR to RGB
plt.show()
