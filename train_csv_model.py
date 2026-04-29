import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("mitbih_train.csv", header=None)

# Split features & labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("ecg_model_new.pkl", "wb"))

print("✅ Model trained & saved successfully!")