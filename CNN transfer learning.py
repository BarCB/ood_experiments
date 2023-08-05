import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
from keras.models import Sequential

BATCH_SIZE = 32
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10

def load_dataset(path:str):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

def optimize_datasets(train_dataset, test_dataset):
    #Configurar el conjunto de datos para el rendimiento
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset

def create_model():
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
  
    model.summary()
    return model;

def train_model(model, train_dataset, validation_dataset=None):
    epochs=10
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )


train_dataset0 = load_dataset(f"C:\\Users\\Barnum\\Desktop\\experiments3\\unlabeled\\MNIST_SVHN_ood0.5_StepFunctionPositive_images100\\batch_9\\test")
train_dataset1 = load_dataset(f"C:\\Users\\Barnum\\Desktop\\experiments3\\unlabeled\\MNIST_SVHN_ood0.5_StepFunctionPositive_images100\\batch_9\\train")

#test_dataset = load_dataset(f"MNIST_SALTANDPEPPER_ood05_sinFiltrar_images200/batch_0/test")
train_dataset, test_dataset = optimize_datasets(train_dataset0, train_dataset1)
model = create_model()
training_result = train_model(model, train_dataset, test_dataset)