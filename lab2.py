import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
iris = load_iris()
print(iris.keys())
x = iris.data
y = iris.target
print(x.shape)
print(y.shape)
print(iris.data[:5])
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
train_accuracy_sgd = []
train_accuracy_adam = []
train_loss_sgd = []
train_loss_adam = []
val_accuracy = []
val_loss = []
for epoch in range(50):
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    for i in range(0, len(X_train), 8):
        X_batch = X_train_shuffled[i:i+8]
        y_batch = y_train_shuffled[i:i+8]
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        sgd.apply_gradients(zip(gradients, model.trainable_variables))
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(zip(gradients, model.trainable_variables))
    train_predictions = model(X_train, training=False)
    train_loss = loss_fn(y_train, train_predictions).numpy()
    train_acc = np.mean(
        np.argmax(train_predictions.numpy(), axis=1) == y_train
    )
    val_predictions = model(X_test, training=False)
    validation_loss = loss_fn(y_test, val_predictions).numpy()
    validation_acc = np.mean(
        np.argmax(val_predictions.numpy(), axis=1) == y_test
    )
    train_loss_sgd.append(train_loss)
    train_loss_adam.append(train_loss)
    train_accuracy_sgd.append(train_acc)
    train_accuracy_adam.append(train_acc)
    val_loss.append(validation_loss)
    val_accuracy.append(validation_acc)
    print(
        f"Epoch {epoch + 1}/50 - "
        f"Loss: {train_loss:.4f} - "
        f"Accuracy: {train_acc:.4f} - "
        f"Validation Loss: {validation_loss:.4f} - "
        f"Validation Accuracy: {validation_acc:.4f}"
    )
model.summary()
train_accuracy = train_accuracy_adam[-1]
val_accuracy_final = val_accuracy[-1]
print("Final Training Accuracy:", train_accuracy)
print("Final Validation Accuracy:", val_accuracy_final)
test_predictions = model(X_test, training=False)
test_loss = loss_fn(y_test, test_predictions).numpy()
test_accuracy = np.mean(
    np.argmax(test_predictions.numpy(), axis=1) == y_test
)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
results = pd.DataFrame({
    'Epoch': range(1, 51),
    'Training Accuracy': train_accuracy_adam,
    'Validation Accuracy': val_accuracy,
    'Training Loss': train_loss_adam,
    'Validation Loss': val_loss
})
print(results)
print("Training Data:")
print(x_train)
print("Training Labels:")
print(y_train)
print("Validation Accuracy:")
print(val_accuracy)
print("Training Accuracy:")
print(train_accuracy_adam)
plt.figure(figsize=(8, 5))
plt.plot(
    train_accuracy_adam,
    label='Training Accuracy'
)
plt.plot(
    val_accuracy,
    label='Validation Accuracy'
)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(
    train_loss_adam,
    label='Training Loss'
)
plt.plot(
    val_loss,
    label='Validation Loss'
)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()