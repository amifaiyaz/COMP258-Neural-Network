import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


with open("C:/Users/amifa/OneDrive/Desktop/COMP 258 - Neural Networks/Assignment 1/hepatitis_training_data.json",'r') as file:
    data = json.load(file)

X = []
y = []
for entry in data:
    X.append([entry[attr] if entry[attr] is not None else np.nanmean([e[attr] for e in data if e[attr] is not None]) for attr in entry if attr != 'Die_Live'])
    y.append([1, 0] if entry['Die_Live'] == 1 else [0, 1])

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(19,)),  # Input layer with 19 neurons
    keras.layers.Dense(30, activation='sigmoid'),  # Second layer with 30 neurons
    keras.layers.Dense(15, activation='sigmoid'),  # Third layer with 15 neurons
    keras.layers.Dense(2, activation='sigmoid')   # Output layer with 2 neurons
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

with open("C:/Users/amifa/OneDrive/Desktop/COMP 258 - Neural Networks/Assignment 1/hepatitis_testing_data.json",'r') as file2:
    testing_data = json.load(file2)
    
  

X_test = []
y_test = []
for entry in testing_data:
    X_test.append([entry[attr] if entry[attr] is not None else np.nanmean([e[attr] for e in testing_data if e[attr] is not None]) for attr in entry if attr != 'Die_Live'])
    y_test.append([1, 0] if entry['Die_Live'] == 1 else [0, 1])


X_test = np.array(X_test)
y_test = np.array(y_test)


X_test = scaler.transform(X_test)

predictions = model.predict(X_test)


predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = ['Die_Live 1' if label == 0 else 'Die_Live 2' for label in predicted_classes]

print("Predicted Labels:")
print(predicted_labels)

correct_predictions = (predicted_classes == np.argmax(y_test, axis=1))
accuracy = np.mean(correct_predictions)
print(f"Accuracy on the testing data: {accuracy * 100:.2f}%")
