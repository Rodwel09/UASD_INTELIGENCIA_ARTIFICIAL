import os
import numpy as np
from keras.api.layers import Dense
from keras.api.models import Sequential

def main():
    # Generate dummy data
    X = np.random.random((1000, 20))
    y = np.random.randint(2, size=(1000, 1))

    # Create a simple neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X, y)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Entrena el modelo con datos aleatorios y evalúa su rendimiento.