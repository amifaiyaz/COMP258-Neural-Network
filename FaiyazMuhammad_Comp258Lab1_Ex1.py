import numpy as np

# Define training patterns and target values
a1 = np.array ([[-1, -1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [ 1, 1, 1, -1, 1, 1, 1]])
a2 = np.array ([[-1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1]])
a3 = np.array ([[-1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, 1, -1, -1, -1, 1, 1]])
b1 = np.array ([[1, 1, 1, 1, 1, 1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [1, 1, 1, 1, 1, 1, -1]])
b2 = np.array ([[1, 1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, 1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, 1, 1, 1, 1, 1, -1]])
b3 = np.array ([[1, 1, 1, 1, 1, 1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [1, 1, 1, 1, 1, 1, -1]])
c1 = np.array ([[-1, -1, 1, 1, 1, 1, 1], [-1, 1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, -1, 1, 1, 1, 1, -1]])
c2 = np.array ([[-1, -1, 1, 1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, 1, -1], [-1, -1, 1, 1, 1, -1, -1]])
c3 = np.array ([[-1, -1, 1, 1, 1, -1, 1], [-1, 1, -1, -1, -1, 1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, 1, -1], [-1, -1, 1, 1, 1, -1, -1]])
d1 = np.array ([[1, 1, 1, 1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, 1, -1], [1, 1, 1, 1, 1, -1, -1]])
d2 = np.array ([[1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, -1, 1, -1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, 1, -1], [1, 1, 1, 1, 1, -1, -1]])
d3 = np.array ([[1, 1, 1, 1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, 1, -1], [1, 1, 1, 1, 1, -1, -1]])
e1 = np.array ([[1, 1, 1, 1, 1, 1, 1], [-1, 1, -1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, 1, 1, -1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, -1, -1, -1, -1, -1], [-1, 1, -1, -1, -1, -1, 1], [1, 1, 1, 1, 1, 1, 1]])
e2 = np.array ([[1, 1, 1, 1, 1, 1, 1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1]])
e3 = np.array ([[1, 1, 1, 1, 1, 1, 1], [-1, 1, -1, -1, -1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, 1, 1, -1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, -1, -1, -1, -1, -1], [-1, 1, -1, -1, -1, -1, -1], [-1, 1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1]])
j1 = np.array ([[-1, -1, -1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, -1, 1, 1, 1, -1, -1]])
j2 = np.array ([[-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, -1, 1, 1, 1, -1, -1]])
j3 = np.array ([[-1, -1, -1, -1, 1, 1, 1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, -1, 1, 1, 1, -1, -1]])
k1 = np.array ([[1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, -1, 1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, 1, -1, -1, -1, -1], [-1, 1, 1, -1, -1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, -1, -1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [1, 1, 1, -1, -1, 1, 1]])
k2 = np.array ([[1, -1, -1, -1, -1, 1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, 1, -1, -1, -1], [1, -1, 1, -1, -1, -1, -1], [1, 1, -1, -1, -1, -1, -1], [1, -1, 1, -1, -1, -1, -1], [1, -1, -1, 1, -1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, -1, 1, -1]])
k3 = np.array ([[1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, -1, -1, 1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, 1, -1, -1, -1, -1], [-1, 1, -1, 1, -1, -1, -1], [-1, 1, -1, -1, 1, -1, -1], [-1, 1, -1, -1, -1, 1, -1], [1, 1, 1, -1, -1, 1, 1]])

# Create a list of all patterns
training_patterns = [a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3, e1, e2, e3, j1, j2, j3, k1, k2, k3]

# Define target values
target_values = [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

# Initialize Perceptron and ADALINE models with random weights and bias
perceptron_weights = np.random.uniform(-1, 1, 63)
perceptron_bias = np.random.uniform(-1, 1)
adaline_weights = np.random.uniform(-1, 1, 63)
adaline_bias = np.random.uniform(-1, 1)

# Define the learning rate for ADALINE
learning_rate = 0.1

# Implement Perceptron training algorithm
def train_perceptron(weights, bias):
    # Initialize a convergence flag
    converged = False
    while not converged:
        converged = True
        for i, pattern in enumerate(training_patterns):
            perceptron_result = classify_input(pattern, weights, bias)
            if perceptron_result != target_values[i]:
                # Update weights and bias
                weights += target_values[i] * pattern.flatten()
                bias += target_values[i]
                converged = False

# Implement ADALINE training algorithm
def train_adaline(weights, bias):
    # Initialize a convergence flag
    converged = False
    while not converged:
        converged = True
        for i, pattern in enumerate(training_patterns):
            adaline_result = classify_input(pattern, weights, bias)
            error = target_values[i] - adaline_result
            if error != 0:
                # Update weights and bias
                weights += learning_rate * error * pattern.flatten()
                bias += learning_rate * error
                converged = False


# Implement classification for the given pattern using Perceptron and ADALINE
def classify_input(pattern, weights, bias):
    weighted_sum = np.dot(pattern.flatten(), weights) + bias
    return 1 if weighted_sum >= 0 else -1

# Main loop for training the Perceptron and ADALINE models
train_perceptron(perceptron_weights, perceptron_bias)
train_adaline(adaline_weights, adaline_bias)

import random

# Function to generate noisy and missing data patterns
def generate_noisy_patterns(patterns, num_errors):
    noisy_patterns = []
    for pattern in patterns:
        noisy_pattern = pattern.copy()
        num_to_change = min(num_errors, len(noisy_pattern))  # Ensure num_to_change is not greater than the length of the pattern
        indices_to_change = random.sample(range(len(pattern)), num_to_change)
        for index in indices_to_change:
            noisy_pattern[index] *= -1  # Flip the pixel value
        noisy_patterns.append(noisy_pattern)
    return noisy_patterns

# Function to classify patterns and print the results
def classify_and_print_results(patterns, perceptron_weights, perceptron_bias, adaline_weights, adaline_bias):
    for pattern in patterns:
        perceptron_result = classify_input(pattern, perceptron_weights, perceptron_bias)
        adaline_result = classify_input(pattern, adaline_weights, adaline_bias)
        print("Pattern:")
        print(pattern)
        print(f"Perceptron Classification: {'B' if perceptron_result == 1 else 'Not B'}")
        print(f"ADALINE Classification: {'B' if adaline_result == 1 else 'Not B'}")
        print()

# Generate and test noisy patterns with different levels of noise
for num_errors in [5, 10, 15, 20]:
    noisy_patterns = generate_noisy_patterns(training_patterns, num_errors)
    print(f"Testing with {num_errors} pixels wrong:")
    classify_and_print_results(noisy_patterns, perceptron_weights, perceptron_bias, adaline_weights, adaline_bias)

# Function to generate missing data patterns
def generate_missing_data_patterns(patterns, num_missing_pixels):
    missing_data_patterns = []
    for pattern in patterns:
        missing_pattern = pattern.copy()
        num_to_remove = min(num_missing_pixels, len(missing_pattern))  # Ensure num_to_remove is not greater than the length of the pattern
        indices_to_remove = random.sample(range(len(pattern)), num_to_remove)
        for index in indices_to_remove:
            missing_pattern[index] = 0  # Set pixel to 0 (missing data)
        missing_data_patterns.append(missing_pattern)
    return missing_data_patterns

# Generate and test missing data patterns with different levels of missing data
for num_missing_pixels in [5, 10, 15, 20]:
    missing_data_patterns = generate_missing_data_patterns(training_patterns, num_missing_pixels)
    print(f"Testing with {num_missing_pixels} pixels missing:")
    classify_and_print_results(missing_data_patterns, perceptron_weights, perceptron_bias, adaline_weights, adaline_bias)

# Function to count classification differences between Perceptron and ADALINE
def count_differences(patterns, perceptron_weights, perceptron_bias, adaline_weights, adaline_bias):
    perceptron_misclassified = 0
    adaline_misclassified = 0
    for pattern in patterns:
        perceptron_result = classify_input(pattern, perceptron_weights, perceptron_bias)
        adaline_result = classify_input(pattern, adaline_weights, adaline_bias)
        if perceptron_result != adaline_result:
            perceptron_misclassified += 1
            adaline_misclassified += 1
    return perceptron_misclassified, adaline_misclassified

# Test patterns with different levels of noise and missing data
for num_errors in [5, 10, 15, 20]:
    noisy_patterns = generate_noisy_patterns(training_patterns, num_errors)
    perceptron_errors, adaline_errors = count_differences(noisy_patterns, perceptron_weights, perceptron_bias, adaline_weights, adaline_bias)
    print(f"With {num_errors} pixels wrong:")
    print(f"Perceptron Misclassified: {perceptron_errors} patterns")
    print(f"ADALINE Misclassified: {adaline_errors} patterns")
    print()

for num_missing_pixels in [5, 10, 15, 20]:
    missing_data_patterns = generate_missing_data_patterns(training_patterns, num_missing_pixels)
    perceptron_errors, adaline_errors = count_differences(missing_data_patterns, perceptron_weights, perceptron_bias, adaline_weights, adaline_bias)
    print(f"With {num_missing_pixels} pixels missing:")
    print(f"Perceptron Misclassified: {perceptron_errors} patterns")
    print(f"ADALINE Misclassified: {adaline_errors} patterns")
    print()

