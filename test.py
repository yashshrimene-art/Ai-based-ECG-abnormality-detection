import tensorflow as tf

model = tf.keras.models.load_model("ecg_image_model.h5", compile=False)
print("MODEL LOADED SUCCESSFULLY ✅")