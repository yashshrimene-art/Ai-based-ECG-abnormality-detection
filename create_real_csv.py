import pandas as pd
import os

# Load dataset (IMPORTANT: check path)
data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\New folder\mitbih_train.csv", header=None)

# Create folder
os.makedirs("real_test_files", exist_ok=True)

# NORMAL files
normal = data[data.iloc[:, -1] == 0].sample(10)
for i in range(10):
    normal.iloc[i:i+1].to_csv(f"real_test_files/normal_{i+1}.csv", index=False, header=False)

# ABNORMAL files
abnormal = data[data.iloc[:, -1] != 0].sample(10)
for i in range(10):
    abnormal.iloc[i:i+1].to_csv(f"real_test_files/abnormal_{i+1}.csv", index=False, header=False)

print("✅ Real test files created!")