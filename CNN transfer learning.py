import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
from keras.models import Sequential
from pathlib import Path

BATCH_SIZE = 32
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10
WEIGHTS_LOCATION = "weights\\"

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
        validation_data = validation_dataset,
        epochs=epochs
    )

def pretraining_model():
    test_dataset = load_dataset(f"C:\\Users\\Barnum\\Desktop\\experiments4\\source\\batch_0\\test")
    train_dataset = load_dataset(f"C:\\Users\\Barnum\\Desktop\\experiments4\\source\\batch_0\\train")

    #test_dataset = load_dataset(f"MNIST_SALTANDPEPPER_ood05_sinFiltrar_images200/batch_0/test")
    train_dataset, test_dataset = optimize_datasets(train_dataset, test_dataset)
    model = create_model()
    
    training_result = train_model(model, train_dataset, test_dataset)
    model.save_weights(WEIGHTS_LOCATION, overwrite=True, save_format=None, options=None)
    
    accuracy = training_result.history['accuracy']
    val_accuracy = training_result.history['val_accuracy']

    loss = training_result.history['loss']
    val_loss = training_result.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def batch_test():
    epochs=10
    accuracies = []
    losses = []
    for number in range(2):
        location_path = f"C:\\Users\\Barnum\\Desktop\\experiments4\\target\\manual GN ood\\batch_{number}\\"
        train_dataset = load_dataset(location_path + "train")
        test_dataset = load_dataset(location_path + "test")
        train_dataset, test_dataset = optimize_datasets(train_dataset, test_dataset)
        model = create_model()
        
        model.load_weights(WEIGHTS_LOCATION, skip_mismatch=False, by_name=False, options=None)

        training_result = train_model(model, train_dataset, test_dataset)
        accuracies.append(training_result.history['val_accuracy'][epochs-1])
        losses.append(training_result.history['val_loss'][epochs-1])
        print(accuracies)

def main():
    #batch_test()   
    pretraining_model()

if __name__ == "__main__":
    main()