# train.py
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# MLflow Setup
mlflow.tensorflow.autolog()
mlflow.set_experiment("Cats_vs_Dogs")

with mlflow.start_run():
    IMG_SIZE = 128
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_gen = train_datagen.flow_from_directory('data/train', target_size=(IMG_SIZE, IMG_SIZE),
                                                  batch_size=BATCH_SIZE, class_mode='binary', subset='training')
    val_gen = train_datagen.flow_from_directory('data/train', target_size=(IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE, class_mode='binary', subset='validation')

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save('models/cat_dog_model.h5')
