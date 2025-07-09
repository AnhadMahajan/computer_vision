import tensorflow as tf
import numpy as np
import pickle

with open("sign_data.pkl", "rb") as f:
    data = pickle.load(f)

labels = list(data.keys())
X, y = [], []

for label in labels:
    for sample in data[label]:
        X.append(sample)
        y.append(labels.index(label))

X = np.array(X)
y = np.array(y)

y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=20, batch_size=16, verbose=1)

model.save("sign_language_model.h5")
print("Model trained and saved as 'sign_language_model.h5'")
