import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load dataset and preprocess
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, to_categorical(data.target), test_size=0.2, random_state=42)

# Create a simple model
model = Sequential([
    Dense(10, input_shape=(4,), activation="relu"),
    Dense(10, activation="relu"),
    Dense(3, activation="softmax")
])

# Compile and train the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, verbose=0)

# Save model to .h5 file
model.save("model.h5")
