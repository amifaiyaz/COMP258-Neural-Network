# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 22:46:47 2023

@author: amifa
"""
import numpy as np
import pandas as pd
import json

# Load the training data from the JSON file
with open("C:/Users/amifa/OneDrive/Desktop/COMP 258 - Neural Networks/Assignment 2/training.json", 'r') as f:
    training_data = json.load(f)
iris_df = pd.DataFrame(training_data)

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network architecture
input_layer_size = 4
hidden_layer_size = 5
output_layer_size = 3

# Initialize the weights with random values
np.random.seed(42)
weights1 = np.random.normal(size=(input_layer_size, hidden_layer_size))
weights2 = np.random.normal(size=(hidden_layer_size, output_layer_size))

# Train the neural network using backpropagation
learning_rate = 0.1
for i in range(10000):
    # Forward pass
    inputs = iris_df.iloc[:, :4].values
    targets = pd.get_dummies(iris_df.iloc[:, 4]).values
    hidden_layer_input = np.dot(inputs, weights1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights2)
    output_layer_output = sigmoid(output_layer_input)
    
    # Backward pass
    error = targets - output_layer_output
    output_layer_error = error * sigmoid_derivative(output_layer_output)
    hidden_layer_error = np.dot(output_layer_error, weights2.T) * sigmoid_derivative(hidden_layer_output)
    
    # Update weights
    weights2 += learning_rate * np.dot(hidden_layer_output.T, output_layer_error)
    weights1 += learning_rate * np.dot(inputs.T, hidden_layer_error)

# Load the testing data from the JSON file
with open("C:/Users/amifa/OneDrive/Desktop/COMP 258 - Neural Networks/Assignment 2/test.json", 'r') as f:
    testing_data = json.load(f)
test_df = pd.DataFrame(testing_data)

# Run the forward pass for the test set
test_inputs = test_df.iloc[:, :4].values
test_targets = pd.get_dummies(test_df.iloc[:, 4]).values

hidden_layer_input = np.dot(test_inputs, weights1)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights2)
output_layer_output = sigmoid(output_layer_input)

# Convert the output to the predicted class
predicted_class = np.argmax(output_layer_output, axis=1)
true_class = np.argmax(test_targets, axis=1)

# Calculate the accuracy
accuracy = np.mean(predicted_class == true_class)

# Print the results
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Predicted class: ", predicted_class)
print("True class: ", true_class)
