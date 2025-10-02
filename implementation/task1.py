# ==============================================
# Task 1: Classical ML with Scikit-learn
# Goal: Train a Decision Tree on the Iris dataset
# ==============================================

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Dataset Loading and Initial Preprocessing
print("--- Starting Task 1: Scikit-learn (Iris) ---")
iris = load_iris()
X = iris.data # Features (sepal/petal length/width)
y = iris.target # Target (species: 0, 1, 2)

# Convert to DataFrame for easier handling and inspection (optional but good practice)
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Preprocessing Step 1: Handle Missing Values
# The Iris dataset is clean, but for completeness, we check and fill/drop here.
# Assuming we would fill missing numeric values with the mean:
# df = df.fillna(df.mean())
print(f"Initial dataset shape: {df.shape}")
print("Missing values check: No missing values in the Iris dataset.")

# Preprocessing Step 2: Encode Labels (The target 'y' is already numeric (0, 1, 2),
# so explicit Label Encoding is not strictly necessary for scikit-learn classifiers,
# but the original text mentioned 'encode labels' so we confirm it is complete.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Encoded classes: {le.classes_}") # Show what 0, 1, 2 map to

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 2. Train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
print("\nDecision Tree Classifier Trained successfully.")

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# 3. Evaluation using Accuracy, Precision, and Recall
# Use average='macro' to calculate metrics for each label and find their unweighted mean.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print("--- Task 1 Complete ---")