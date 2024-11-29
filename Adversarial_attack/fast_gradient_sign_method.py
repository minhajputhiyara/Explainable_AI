import tensorflow as tf
import numpy as np

# Load a pre-trained model (e.g., ResNet50)
model = tf.keras.applications.ResNet50(weights="imagenet")
model.trainable = False  # Ensure the model is not trainable

# Load and preprocess an input image (skipping preprocessing steps here)
IMAGE_PATH = "./example.jpg"  # Replace with your image path
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = tf.keras.applications.resnet50.preprocess_input(img)

# Define the target class for the attack (example: "tabby cat" with class index 281)
target_class = 281
target_label = tf.one_hot(target_class, model.output_shape[-1])
target_label = tf.reshape(target_label, (1, -1))

# FGSM Attack
epsilon = 0.01  # Perturbation magnitude

with tf.GradientTape() as tape:
    tape.watch(img)  # Watch the input image for gradient calculation
    predictions = model(img)
    loss = tf.keras.losses.categorical_crossentropy(target_label, predictions)

# Compute gradients
gradients = tape.gradient(loss, img)

# Apply FGSM perturbation
perturbations = epsilon * tf.sign(gradients)
adversarial_image = img + perturbations
adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)  # Clip values to valid range

# Predict using the adversarial image
adversarial_predictions = model(adversarial_image)
predicted_class = tf.argmax(adversarial_predictions[0])

print(f"Original Class: {tf.argmax(predictions[0])}")
print(f"Adversarial Class: {predicted_class}")
