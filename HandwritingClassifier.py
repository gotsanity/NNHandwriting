import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#get the data set
mnist = tf.keras.datasets.mnist

#unpack the dataset to the following (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#do this after showing the values - normalizes the values between 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#lets build a model
model = tf.keras.models.Sequential()
#this flattens out the input model from a multidimensional array to a flat array
model.add(tf.keras.layers.Flatten())
#adding our hidden layers (128 neurons, activation function rectified linear)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#adding our output layer, ensure neurons match possibilities (10) and activation is switch to something that works on probability (softmax)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#model is done. setup the training params of the model
#optimizer: lots of choice here. adam is the go to
#loss: sparse categorical crossentropy is a goto but once again lots of choice. binary for yes/no true false problems
#metrics: what we want to track (in our case just accuracy for now)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#lets train
# model.fit(x_train, y_train, epochs=3)

# #lets calculate loss and accuracy and evalute it
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)

# #to save a model
# model.save('HandwrittenDigits_v01.model')

#to load a model
new_model = tf.keras.models.load_model('HandwrittenDigits_v01.model')

#to make a prediction
predictions = new_model.predict([x_test])
print(predictions)


print (np.argmax(predictions[1]))
plt.imshow(x_test[1], cmap= plt.cm.binary)
plt.show()

#display below


#looking at the array
#print (x_test[0])

#to show what we are looking at lets import matplotlib and plot the array

#plt.imshow(x_train[0], cmap= plt.cm.binary) # show with and without cmap
#plt.show()
