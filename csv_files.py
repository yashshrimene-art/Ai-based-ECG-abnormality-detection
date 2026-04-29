import pandas as pd
import numpy as np
import os

# Create folder
os.makedirs("ecg_test_files", exist_ok=True)

# Generate NORMAL files (label = 0)
for i in range(10):
    data = np.random.rand(1, 187)
    data = np.append(data, 0).reshape(1, -1)
    pd.DataFrame(data).to_csv(f"ecg_test_files/normal_{i+1}.csv", index=False, header=False)

# Generate ABNORMAL files (label = 1)
for i in range(10):
    data = np.random.rand(1, 187)
    data = np.append(data, 1).reshape(1, -1)
    pd.DataFrame(data).to_csv(f"ecg_test_files/abnormal_{i+1}.csv", index=False, header=False)

print("✅ 20 CSV files created successfully!")