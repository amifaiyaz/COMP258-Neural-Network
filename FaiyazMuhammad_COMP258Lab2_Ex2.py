# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:30:49 2023

@author: amifa
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reducing the number of training samples
sample_size = 5000  
x_train = x_train[:sample_size]
y_train = y_train[:sample_size]

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Function to create different MLP architectures
def create_model_1(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_model_2(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Adjusted list of architectures
architectures = [create_model_1((32, 32, 3), 10), create_model_2((32, 32, 3), 10)]

learning_algorithms = ['stochastic', 'batch', 'mini-batch']
learning_parameters = [{'lr': 0.01, 'batch_size': 32}, {'lr': 0.001, 'batch_size': 64}, {'lr': 0.0001, 'batch_size': 128}]

for i in range(len(architectures)):
    model = architectures[i]
    algorithm = learning_algorithms[i % 3]
    parameters = learning_parameters[i % 3]

    if algorithm == 'stochastic':
        optimizer = SGD(learning_rate=parameters['lr'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, batch_size=1)

    elif algorithm == 'batch':
        optimizer = SGD(learning_rate=parameters['lr'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, batch_size=len(x_train))

    elif algorithm == 'mini-batch':
        optimizer = SGD(learning_rate=parameters['lr'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, batch_size=parameters['batch_size'])

    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Architecture {i+1}, Algorithm: {algorithm}, Accuracy: {accuracy}")

    # Save the model
    model.save(f'model_{i+1}.h5')

# Making predictions
predictions = model.predict(x_test)

# Model evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")

# Printing the first 10 predictions
for i in range(10):
    print(f"Predicted class for sample {i + 1}: {predictions[i].argmax()}")