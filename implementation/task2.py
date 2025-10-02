# ==============================================
# Task 2: Deep Learning with TensorFlow (Keras)
# Goal: Build a CNN for MNIST, achieve >95% accuracy
# ==============================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

print("--- Starting Task 2: TensorFlow CNN (MNIST) ---")

# 1. Dataset Loading and Preprocessing
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing: Reshape data to include a single color channel (required for CNN)
# MNIST images are 28x28 grayscale. New shape: (samples, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Preprocessing: Normalize pixel values to the range [0, 1]
X_train /= 255
X_test /= 255

# Preprocessing: Convert target labels to one-hot encoding
# (e.g., digit '5' becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)
print("Data loaded, reshaped, normalized, and one-hot encoded.")

# 2. Build a CNN Model
model = Sequential([
    # First Conv Layer: Extract features
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    # Second Conv Layer: More features
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), # Dropout for regularization
    # Transition to Dense Layers
    Flatten(),
    # Dense Layers for classification
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') # Output layer (10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nModel Architecture:")
model.summary()

# 3. Training the Model
# Use fewer epochs for a quick run; 10-15 epochs typically ensures >95% accuracy.
epochs = 10
batch_size = 128
history = model.fit(X_train, y_train_onehot,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test_onehot))

# 4. Final Evaluation
score = model.evaluate(X_test, y_test_onehot, verbose=0)
test_accuracy = score[1]
print(f"\n--- Model Evaluation ---")
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} (Goal: >0.95)")

if test_accuracy > 0.95:
    print("SUCCESS: Achieved greater than 95% test accuracy!")
else:
    print("Note: Accuracy goal not met. Try increasing epochs or tuning the model.")


# 5. Visualize Model's Predictions on 5 Sample Images
print("\n--- Visualizing Predictions on 5 Samples ---")
# Select 5 random images from the test set
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_images = X_test[sample_indices]
sample_labels = y_test[sample_indices]

# Predict
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)

# Plotting
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle('CNN Predictions on Sample Images', fontsize=16)

for i in range(5):
    axes[i].imshow(sample_images[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {sample_labels[i]}\nPred: {predicted_classes[i]}")
    axes[i].axis('off')

plt.show() #
print("--- Task 2 Complete ---")