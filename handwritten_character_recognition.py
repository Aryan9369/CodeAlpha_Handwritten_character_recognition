import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load and Prepare the MNIST Dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"Original shapes: x_train={x_train.shape}, y_train={y_train.shape}, x_test={x_test.shape}, y_test={y_test.shape}")

# --- Data Exploration (Optional) ---
# plt.figure(figsize=(10, 5))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(x_train[i], cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# --- Preprocessing ---
# Scale images to the [0, 1] range (Normalization)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Make sure images have shape (28, 28, 1) for CNN input
# Add a channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(f"Shapes after adding channel: x_train={x_train.shape}, x_test={x_test.shape}")

# Convert class vectors to binary class matrices (One-Hot Encoding)
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"Example original label: {y_train[0]}")
print(f"Example one-hot encoded label: {y_train_cat[0]}")

# 2. Build the CNN Model
input_shape = (28, 28, 1)

print("\nBuilding CNN model...")
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), # 32 filters, 3x3 kernel
        layers.MaxPooling2D(pool_size=(2, 2)), # Downsample
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), # 64 filters
        layers.MaxPooling2D(pool_size=(2, 2)), # Downsample again
        layers.Flatten(), # Convert 2D feature maps to 1D vector
        layers.Dropout(0.5), # Regularization to prevent overfitting
        layers.Dense(num_classes, activation="softmax"), # Output layer: 10 neurons (0-9), softmax for probabilities
    ]
)

model.summary() # Print model architecture

# 3. Compile the Model
print("\nCompiling model...")
# Adam is a good general-purpose optimizer
# categorical_crossentropy is standard for multi-class classification with one-hot labels
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 4. Train the Model
batch_size = 128 # Number of samples per gradient update
epochs = 15     # Number of passes over the entire training dataset

print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")

# Use a portion of training data for validation during training
history = model.fit(
    x_train,
    y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1 # Use 10% of training data for validation
)

print("\nTraining complete.")

# 5. Evaluate the Model on the Test Set
print("\nEvaluating model on test set...")
score = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# --- Plot training history ---
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()


# 6. Generate Predictions and Detailed Report
print("\nGenerating predictions on test set...")
y_pred_proba = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1) # Get class index with highest probability

print("\nClassification Report:")
# Use original y_test (0-9) for classification report comparison
print(classification_report(y_test, y_pred_classes))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# --- Visualize Some Predictions (Optional) ---
# plt.figure(figsize=(12, 12))
# for i in range(25): # Show first 25 test images
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28), cmap='gray') # Reshape back to 2D for plotting
#     true_label = y_test[i]
#     pred_label = y_pred_classes[i]
#     color = 'green' if pred_label == true_label else 'red'
#     plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# 7. Save the Model (Optional)
# model.save("mnist_cnn_model.h5")
# print("\nModel saved as mnist_cnn_model.h5")

# To load later:
# loaded_model = keras.models.load_model("mnist_cnn_model.h5")
