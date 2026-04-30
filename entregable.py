import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

digits = load_digits()
x = digits.images
y = digits.target

x = x.astype('float32') / 255.0
x = x[..., np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(f'Numero: {y_train[i]}')
    plt.axis('off')
plt.suptitle('Muestra del dataset load_digits (multiclase)')
plt.tight_layout()
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(8, 8, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print("Precisión:", acc)

y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de confusion')
plt.show()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f'Real:{y_test[i]} Pred:{y_pred[i]}')
    plt.axis('off')
plt.suptitle('Predicciones vs etiquetas reales')
plt.tight_layout()
plt.show()

pred = model.predict(x_test[:1])
print("Prediccion individual:", np.argmax(pred))
print("Real:", y_test[0])