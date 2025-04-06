import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split # <<< CHANGE >>> For splitting custom data
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os #  To interact with file system
import pathlib #  For easier path handling

#  Define parameters for your custom dataset
# 
dataset_path = '/kaggle/input/mnistasjpg' # 
img_height = 28  # Target image height (choose a size, e.g., 28 or 32)
img_width = 28   # Target image width
num_channels = 1 # 1 for grayscale, 3 for RGB (adjust if needed)
color_mode = 'grayscale' if num_channels == 1 else 'rgb'
validation_split_ratio = 0.2 # Use 20% of data for testing

# 1. Load Custom Dataset
print(f"Loading custom dataset from: {dataset_path}")
data_dir = pathlib.Path(dataset_path)

# Check if directory exists
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found at {data_dir}. Please check the path.")

image_paths = list(data_dir.glob('*/*')) # Get all image paths in subfolders
image_count = len(image_paths)
print(f"Found {image_count} images.")

if image_count == 0:
     raise ValueError(f"No images found in subdirectories of {data_dir}. Check your folder structure.")

# Get class names from folder names
class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

if num_classes == 0:
     raise ValueError(f"No class subdirectories found in {data_dir}. Check your folder structure.")

# Create a mapping from class name to integer index
class_map = {name: i for i, name in enumerate(class_names)}

# Load images and labels manually
all_images = []
all_labels_int = [] # Store integer labels first

print("Reading image files...")
for class_name in class_names:
    class_dir = data_dir / class_name
    for image_path in class_dir.glob('*'):
        try:
            # Load image, convert to target size and color mode
            img = keras.preprocessing.image.load_img(
                image_path,
                target_size=(img_height, img_width),
                color_mode=color_mode
            )
            # Convert image to numpy array
            img_array = keras.preprocessing.image.img_to_array(img)
            all_images.append(img_array)
            all_labels_int.append(class_map[class_name])
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Error: {e}")

if not all_images:
    raise ValueError("Failed to load any images. Check image files and paths.")

# Convert lists to NumPy arrays
x_data = np.array(all_images)
y_data_int = np.array(all_labels_int)
print(f"Loaded data shapes: x_data={x_data.shape}, y_data_int={y_data_int.shape}")

# --- Preprocessing ---
# Scale images to the [0, 1] range (Normalization)
x_data = x_data.astype("float32") / 255.0

# Ensure channel dimension is correct (already handled by load_img and img_to_array if needed)
# If grayscale and shape is (n, height, width), add channel dim:
# if num_channels == 1 and len(x_data.shape) == 3:
#     x_data = np.expand_dims(x_data, -1)
# print(f"Data shape after potential channel expansion: {x_data.shape}")


# Split data into training and testing sets
print(f"Splitting data into training and testing sets (Test ratio: {validation_split_ratio})...")
x_train, x_test, y_train_int, y_test_int = train_test_split(
    x_data,
    y_data_int,
    test_size=validation_split_ratio,
    random_state=42,  # for reproducibility
    stratify=y_data_int # Ensure class distribution is similar in train/test
)

# Convert class vectors to binary class matrices (One-Hot Encoding) AFTER splitting
y_train_cat = keras.utils.to_categorical(y_train_int, num_classes)
y_test_cat = keras.utils.to_categorical(y_test_int, num_classes)

print(f"Final shapes: x_train={x_train.shape}, y_train_cat={y_train_cat.shape}, x_test={x_test.shape}, y_test_cat={y_test_cat.shape}")


# 2. Build the CNN Model
# Use image dimensions and number of classes from custom data
input_shape = (img_height, img_width, num_channels)

print("\nBuilding CNN model...")
model = keras.Sequential(
    [
        keras.Input(shape=input_shape), #  Adapted input shape
        # --- You might need to adjust the CNN architecture based on image size/complexity ---
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Add more Conv/Pooling layers if needed for larger images or more complex chars
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"), #  Adapted output neurons
    ]
)

model.summary() # Print model architecture

# 3. Compile the Model
print("\nCompiling model...")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 4. Train the Model
batch_size = 32 # Adjust as needed, smaller batches might be better for smaller datasets
epochs = 20     # Adjust as needed, more epochs might be required

print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")

# Train using the custom data
# Use validation_split on the TRAINING data passed to fit
history = model.fit(
    x_train,
    y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1 # Use 10% of training data (x_train) for validation DURING training
)

print("\nTraining complete.")

# 5. Evaluate the Model on the Test Set
print("\nEvaluating model on the held-out test set...")
# Evaluate using the dedicated test set (x_test, y_test_cat)
score = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# --- Plot training history ---
plt.figure(figsize=(12, 5))
# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
# Check if validation accuracy exists in history (it should if validation_split > 0)
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
if 'val_loss' in history.history:
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
y_pred_classes = np.argmax(y_pred_proba, axis=1) # Get predicted integer class index

print("\nClassification Report:")
# Use integer test labels (y_test_int) and provide class names
#  Use custom class_names
#  Add the 'labels' parameter and 'zero_division'
print(classification_report(
    y_test_int,
    y_pred_classes,
    labels=range(num_classes),  # Explicitly list all possible class indices (0, 1, 2, 3)
    target_names=class_names,   # The names corresponding to those indices
    zero_division=0             # Handles cases where a class might have no true samples in the test set, preventing warnings/errors
))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_int, y_pred_classes)
plt.figure(figsize=(max(8, num_classes // 2), max(6, num_classes // 2.5))) # Adjust figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names) #  Add class names to axes
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Optional: Visualize Some Predictions - Adapt if needed based on data shape
# print("\nVisualizing some predictions...")
# plt.figure(figsize=(12, 12))
# num_images_to_show = min(25, len(x_test)) # Show up to 25 images
# for i in range(num_images_to_show):
#     plt.subplot(5, 5, i + 1)
#     # Adjust plotting based on whether images are grayscale or RGB
#     if num_channels == 1:
#         plt.imshow(x_test[i].reshape(img_height, img_width), cmap='gray')
#     else:
#         plt.imshow(x_test[i]) # Assumes pixel values are already 0-1
#     true_label = class_names[y_test_int[i]]
#     pred_label = class_names[y_pred_classes[i]]
#     color = 'green' if pred_label == true_label else 'red'
#     plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
#     plt.axis('off')
# plt.tight_layout()
# plt.show()


# 7. Save the Model (Optional)
# model_filename = "custom_hcr_model.h5"
# model.save(model_filename)
# print(f"\nModel saved as {model_filename}")
