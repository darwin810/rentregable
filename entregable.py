import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]


for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(f'Dígito: {y_train[i]}')
    plt.axis('off')
plt.suptitle('Muestra del dataset MNIST (multiclase 0-9)')
plt.tight_layout()
plt.show()


x_simple = np.array([[5],[7],[10],[11],[12],[15],[16],[18]])
y_simple  = np.array([0, 0,  0,   1,   1,   2,   2,   2])

modelo_simple = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(63, activation='relu'),
    tf.keras.layers.Dense(3,  activation='softmax')
])
modelo_simple.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
modelo_simple.fit(x_simple, y_simple, epochs=1000, verbose=0)

pred_simple = np.argmax(modelo_simple.predict(np.array([[18]])), 1)
print('Predicción clase (valor 18):', pred_simple)


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


model.fit(
    x_train, y_train,
    epochs=6,
    batch_size=32,
    validation_data=(x_test, y_test)
)


loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nPrecisión en test: {acc:.4f}')

y_pred = np.argmax(model.predict(x_test), axis=1)

print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred,
      target_names=[str(i) for i in range(10)]))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de confusión — CNN MNIST')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    plt.title(f'R:{y_test[i]} P:{y_pred[i]}', color=color, fontsize=9)
    plt.axis('off')
plt.suptitle('Predicciones vs etiquetas reales (verde=correcto, rojo=error)')
plt.tight_layout()
plt.show()

# ── 9. PREDICCIÓN INDIVIDUAL ─────────────────────────────────────────────────

pred = model.predict(x_test[:1])
print('Predicción:', np.argmax(pred))
print('Real:      ', y_test[0])