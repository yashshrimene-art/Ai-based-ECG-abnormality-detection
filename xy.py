import kagglehub

# Download latest version
path = kagglehub.dataset_download("erhmrai/ecg-image-data")

print("Path to dataset files:", path)