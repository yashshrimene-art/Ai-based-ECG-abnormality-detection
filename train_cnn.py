# modelcnn.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import pickle

print("🚀 Loading dataset...")

train_dir = r"C:\Users\Lenovo\.cache\kagglehub\datasets\erhmrai\ecg-image-data\versions\1\ECG_Image_data\train"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Save class mapping
class_indices = train_generator.class_indices
print("Class Indices:", class_indices)

with open("class_indices.pkl", "wb") as f:
    pickle.dump(class_indices, f)

# CNN MODEL
model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(len(class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("⏳ Training model...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# 🔥 IMPORTANT FIX: Save in .keras format
model.save("ecg_image_model.keras")

print("✅ Model saved successfully!")