import numpy as np
from keras.api.layers import Dense
from keras.api.models import Sequential

# Datos de entrenamiento para la compuerta XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")
# Crear el modelo
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
# Entrenar el modelo
model.fit(training_data, target_data, epochs=1000)
# Evaluar el modelo
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Predecir con el modelo
print(model.predict(training_data).round())
# Este código crea una red neuronal simple con una capa oculta de 16 neuronas y una capa de salida de una neurona1. 
# La red se entrena para aprender la función XOR, que es un problema clásico en el aprendizaje automático.