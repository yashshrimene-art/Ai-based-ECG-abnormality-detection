# model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

print("🚀 Loading dataset...")

# Load dataset
data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\New folder\mitbih_train.csv", header=None)

# Reduce size for faster training
data = data.sample(30000, random_state=42)

print("✅ Dataset loaded")

# Features & Labels
X = data.iloc[:, :-1].astype(float)
y = data.iloc[:, -1]

# Convert multi-class → binary
y = y.apply(lambda x: 0 if x == 0 else 1)

print("\n📊 Label Distribution:")
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    class_weight={0: 1, 1: 3},
    random_state=42,
    n_jobs=-1
)

print("\n⏳ Training model...")
model.fit(X_train, y_train)

print("✅ Training completed!")

# Accuracy
print("\n📈 Train Accuracy:", model.score(X_train, y_train))
print("📈 Test Accuracy:", model.score(X_test, y_test))

# Save model
pickle.dump(model, open("ecg_model.pkl", "wb"))

print("💾 Model saved as ecg_model.pkl")