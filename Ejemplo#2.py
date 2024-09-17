import os
import numpy as np
from keras.api.layers import Dense
from keras.api.models import Sequential

X = np.random.random((1000, 20))
y = np.random.randint(2, size=(1000, 1))

# Crear un modelo de red neuronal simple
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=10, batch_size=32)

# Evaluar el modelo
loss, accuracy = model.evaluate(X, y)
print(f'Pérdida: {loss}, Precisión: {accuracy}')