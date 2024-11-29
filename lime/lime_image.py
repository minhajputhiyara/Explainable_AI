import numpy as np
from tensorflow.keras.applications import resnet50
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Load pre-trained model
model = resnet50.ResNet50(weights="imagenet")

# Load and preprocess an image
img_path = './cat.jpg'  # Replace with your image path
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = resnet50.preprocess_input(img_array)

# Create LIME image explainer
explainer = LimeImageExplainer()

# Explain the prediction
explanation = explainer.explain_instance(
    image=img_array[0],
    classifier_fn=lambda x: model.predict(x), 
    top_labels=1,
    hide_color=0,
    num_samples=1000  # Number of perturbations
)

# Visualize explanation
from skimage.color import gray2rgb  # For grayscale images
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    hide_rest=False,
    num_features=5,
    min_weight=0.01
)
plt.imshow(mark_boundaries(gray2rgb(temp) / 255.0, mask))
plt.title("LIME Explanation")
plt.show()
