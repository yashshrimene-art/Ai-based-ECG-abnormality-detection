import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("mitbih_train.csv", header=None)

# Split
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Normalize properly
X = X / 255.0

# Train strong model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X, y)

# Save model
pickle.dump(model, open("ecg_model_final.pkl", "wb"))

print("✅ Model trained successfully!")