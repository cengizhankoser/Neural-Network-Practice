import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""
plt.figure(figsize=(28,28))

for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.axis('off')
plt.show()


#print(train_images.shape)


model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_data=(test_images, test_labels ))
"""

#print(history.history['accuracy'])
#print(history.history['val_accuracy'])
#test_loss, test_accuracy = model.evaluate(test_images, test_labels)


#model.save('model')
loaded_model = load_model('model')
test_loss, test_accuracy = loaded_model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')