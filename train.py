import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Parameters
image_size = 224
batch_size = 32
epochs = 20

# Paths to the dataset
dataset_path = "/Users/srisathvig/Documents/Projects/5 - AI SECURITY/Face Mask Dataset"  # Change to your dataset directory

# Load the dataset
categories = ["with_mask", "without_mask"]
data = []
labels = []

for category in categories:
    path = os.path.join(dataset_path, category)
    class_num = categories.index(category)
    
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized_img = cv2.resize(img_array, (image_size, image_size))
            data.append(resized_img)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img}: {e}")

# Convert data and labels into numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Convert labels into binary format
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Data Augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load MobileNetV2 and freeze the base layers
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

base_model.trainable = False  # Freeze base model

# Build the custom model
head_model = base_model.output
head_model = GlobalAveragePooling2D()(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs
)

# Save the model
model.save("mask_detector_model.h5")

# Evaluate the model
predY = model.predict(testX, batch_size=batch_size)
predY = np.argmax(predY, axis=1)
testY = np.argmax(testY, axis=1)

print(classification_report(testY, predY, target_names=categories))

# Plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()