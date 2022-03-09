import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import  Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(0)

# ======= Prepare Dataset =======
# Download the Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10

# Convert our labels to be Categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Normalize the Data
x_train = x_train / 255
x_test = x_test / 255

# Shape the Data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# ======= Create the Model =======
model = Sequential()
# 	Input Layer
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
#	1st Layer
model.add(Dense(units=128, activation='relu'))
#		Dropout layer (ignores a random portion of previous layer)
model.add(Dropout(0.2))
# 	Output Layer
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ======= Train the Model =======
batch_size = 512
epochs = 10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

# ======= Evaluate =======
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy {test_acc}')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Display a random Sample and the models predicion vs true value
random_i = np.random.choice(len(x_test))
x_sample = x_test[random_i]
y_true = np.argmax(y_test, axis = 1)
y_sample_true = y_true[random_i]
y_sample_pred_class = y_pred_classes[random_i]
plt.title(f'Predicted:{y_sample_pred_class}, True:{y_sample_true}')
plt.imshow(x_sample.reshape(28,28), cmap='gray')

# Create and show a confusion Matrix of the predictions vs true value
confusion_matrix = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')