from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

import os
import numpy as np


# Function to load and preprocess actor face images
def load_and_preprocess_faces(base_folder, target_size=(224, 224)):
    faces = []
    labels = []
    num_folders = 0
    total_images = 0

    actors = sorted(os.listdir(base_folder))
    for label, actor_name in enumerate(actors):
        actor_folder = os.path.join(base_folder, actor_name)
        if os.path.isdir(actor_folder):
            num_folders += 1
            actor_images = os.listdir(actor_folder)
            total_images += len(actor_images)

            for filename in actor_images:
                img_path = os.path.join(actor_folder, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):
                    face = Image.open(img_path)
                    face = face.convert("RGB")
                    face = face.resize(target_size)
                    face = np.array(face) / 255.0
                    faces.append(face)
                    labels.append(label)

    print(f"Number of folders loaded: {num_folders}")
    print(f"Total number of images loaded: {total_images}")

    return np.array(faces), np.array(labels)


# Path to the folder containing actor face images
base_folder = "Coursework/TaskC/CelebrityImages"

# Preprocess actor face images
actor_faces, labels = load_and_preprocess_faces(base_folder)

# Train-validation-test split
train_faces, test_faces, train_labels, test_labels = train_test_split(
    actor_faces, labels, test_size=0.2, random_state=42
)
train_faces, val_faces, train_labels, val_labels = train_test_split(
    train_faces, train_labels, test_size=0.2, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(train_faces)

# Load pre-trained VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
output_classes = len(np.unique(train_labels)) + 1
model = keras.Sequential(
    [
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_classes, activation="softmax"),
    ]
)

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
history = model.fit(
    datagen.flow(train_faces, train_labels, batch_size=32),
    epochs=10,
    validation_data=(val_faces, val_labels),
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_faces, test_labels)
print("\nTest accuracy:", test_acc)

actor_labels = sorted(os.listdir(base_folder))
with open("actor_labels.txt", "w") as f:
    for label in actor_labels:
        f.write(label + "\n")

model.save("MovieActorClassification.h5")

# Predictions on unseen data
unseen_folder = "Coursework/TaskC/UnseenActors"
unseen_faces, unseen_labels = load_and_preprocess_faces(unseen_folder)
predictions = model.predict(unseen_faces)

# Get the predicted labels and corresponding probabilities
predicted_labels = np.argmax(predictions, axis=1)
predicted_probabilities = np.max(predictions, axis=1)

# Sort actor names
actor_names = sorted(os.listdir(base_folder))

# Display predicted labels and probabilities along with actor names
for i in range(len(predicted_labels)):
    predicted_label = predicted_labels[i]
    predicted_probability = predicted_probabilities[i]
    actor_name = actor_names[
        predicted_label
    ]  # Use sorted actor names to get correct actor
    image_name = os.listdir(unseen_folder)[i]  # Get the image name
    print("Image:", image_name)
    print("Predicted actor:", actor_name)
    print("Probability:", predicted_probability)
    print()

# Sort unseen labels based on the sorted actor names
sorted_unseen_labels = [actor_names.index(actor) for actor in os.listdir(unseen_folder)]

# Calculate accuracy using sorted labels
accuracy = np.mean(predicted_labels == sorted_unseen_labels)
print("Unseen data accuracy:", accuracy)
