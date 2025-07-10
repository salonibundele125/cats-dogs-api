import tensorflow as tf
from app.utils import read_image_from_bytes

model = tf.keras.models.load_model("models/cat_dog_model.h5")

def predict_image(image_bytes):
    image_array = read_image_from_bytes(image_bytes)
    prediction = model.predict(image_array)[0][0]
    return "dog" if prediction > 0.5 else "cat"
