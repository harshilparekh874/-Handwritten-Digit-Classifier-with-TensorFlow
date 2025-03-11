#!/usr/bin/env python
# coding: utf-8
"""
HandrwittenDigitAnalyzer

This script loads the MNIST dataset, preprocesses the data,
creates and trains a neural network model, saves the trained model,
evaluates the performance, and visualizes the predictions.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def main():
    # Print TensorFlow version
    print('Using TensorFlow version', tf.__version__)

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Display shapes of the dataset arrays
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    # Plot an example image from the training set
    plt.imshow(x_train[0], cmap='binary')
    plt.title('Example of a Hand-Written Digit')
    plt.show()

    # Display label of the first image and unique labels in the training set
    print('Label of the first image:', y_train[0])
    print('Unique labels in training set:', set(y_train))

    # One Hot Encoding of labels
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # Validate shapes of the encoded labels and display one example
    print('y_train_encoded shape:', y_train_encoded.shape)
    print('y_test_encoded shape:', y_test_encoded.shape)
    print('Encoded label for the first image:', y_train_encoded[0])

    # Reshape the images from 28x28 to 784-element vectors
    x_train_reshaped = np.reshape(x_train, (60000, 784))
    x_test_reshaped = np.reshape(x_test, (10000, 784))
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # Display unique pixel values of the first reshaped training image
    print('Unique pixel values in the first training image:', set(x_train_reshaped[0]))

    # Normalize the data by subtracting the mean and dividing by the standard deviation
    x_mean = np.mean(x_train_reshaped)
    x_std = np.std(x_train_reshaped)
    epsilon = 1e-10  # small constant to avoid division by zero

    x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
    x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

    # Display unique normalized pixel values for the first training image
    print('Unique normalized pixel values in the first training image:', set(x_train_norm[0]))

    # Create the neural network model using a Sequential API
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model with stochastic gradient descent optimizer and categorical crossentropy loss
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print the model summary
    model.summary()

    # Train the model for 3 epochs
    model.fit(x_train_norm, y_train_encoded, epochs=3)

    # Evaluate the model on the test set and print accuracy
    loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
    print('Test set accuracy:', accuracy * 100)

    # Save the trained model for later use
    model.save('mnist_model.h5')
    print('Model saved as mnist_model.h5')

    # Make predictions on the test set
    preds = model.predict(x_test_norm)
    print('Shape of predictions:', preds.shape)

    # Plot a 5x5 grid of test images with prediction results
    plt.figure(figsize=(12, 12))
    start_index = 0
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        # Determine prediction and ground truth for each image
        pred = np.argmax(preds[start_index + i])
        gt = y_test[start_index + i]
        # Color code: green if prediction is correct, red otherwise
        col = 'g' if pred == gt else 'r'
        plt.xlabel(f'i={start_index+i}, pred={pred}, gt={gt}', color=col)
        plt.imshow(x_test[start_index + i], cmap='binary')
    plt.show()

    # Plot the prediction probabilities for the 9th test image (index 8)
    plt.plot(preds[8])
    plt.title('Prediction Probabilities for Test Image 8')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.show()

    model.save('mnist_model.h5')


if __name__ == '__main__':
    main()
