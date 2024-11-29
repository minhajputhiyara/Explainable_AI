import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to apply a grey patch to an image
def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size,
                  top_left_x:top_left_x + patch_size, :] = 127.5
    return patched_image

# Load the image
IMAGE_PATH = './cat.jpg'  # Replace with the actual image path
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)

# Normalize the image for ResNet50
img = tf.keras.applications.resnet50.preprocess_input(img)

# Instantiate the model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
CAT_CLASS_INDEX = 281  # ImageNet index for "tabby cat"
PATCH_SIZE = 40  # Size of the occlusion patch

# Initialize the sensitivity map
sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

# Iterate the patch over the image
for top_left_x in range(0, img.shape[1], PATCH_SIZE):
    for top_left_y in range(0, img.shape[0], PATCH_SIZE):
        # Apply the grey patch
        patched_image = apply_grey_patch(img, top_left_x, top_left_y, PATCH_SIZE)
        patched_image = np.expand_dims(patched_image, axis=0)  # Add batch dimension

        # Predict using the model
        predicted_classes = model.predict(patched_image)[0]
        confidence = predicted_classes[CAT_CLASS_INDEX]

        # Update the sensitivity map
        sensitivity_map[top_left_y:top_left_y + PATCH_SIZE,
                        top_left_x:top_left_x + PATCH_SIZE] = confidence

# Plot the sensitivity map
plt.figure(figsize=(10, 10))
plt.imshow(sensitivity_map, cmap='hot', interpolation='nearest')
plt.title("Occlusion Sensitivity Map")
plt.colorbar(label='Class Confidence')
plt.show()
