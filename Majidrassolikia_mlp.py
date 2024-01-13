import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


with open('train.p', 'rb') as f:
    train_data = pickle.load(f)
train_data = [(np.array(x) / 255.0, y) for x, y in train_data]
np.random.shuffle(train_data)
X = np.array([x for x, y in train_data])
y = np.array([y for x, y in train_data])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)


with open('test.p', 'rb') as f:
    test_data = pickle.load(f)
test_data = [(np.array(x) / 255.0, y) for x, y in test_data]
X_test = np.array([x for x, y in test_data])
y_test = np.array([y for x, y in test_data])


X_test = scaler.transform(X_test)


loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)