import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# get a dataset
mnist = tf.keras.datasets.mnist

#unpacks our data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#normalize the data for easy calculations
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


#make a model
model = tf.keras.models.Sequential()

#flatten the model into a simple 2d array
model.add(tf.keras.layers.Flatten())

#begin adding hidden layers (128 neurons, activation function rectified linear function)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#add our output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#model done, lets compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train
# model.fit(x_train, y_train, epochs=3)

#lets evaluate our accuracy and loss
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)


#to save a model
# model.save('HW_Classifier.model')


#load a model
new_model = tf.keras.models.load_model('HW_Classifier.model')


# make a prediction
predictions = new_model.predict([x_test])
print(predictions)

# to make our predictions human interactable
print(np.argmax(predictions[1]))
plt.imshow(x_test[1], cmap= plt.cm.binary)
plt.show()


#display the data
# print(x_test[0])

# plt.imshow(x_test[0], cmap = plt.cm.binary)
# plt.show()