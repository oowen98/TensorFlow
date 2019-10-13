
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

#Simple model that classifies clothing items


data = keras.datasets.fashion_mnist #dataset from keras

#split dataset into training and testing
#load_data will return the data in the 4 different datasets using Keras, otherwise we will need to split it ourserlves manually
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#class labels from TF
class_names = ['T-shirt.top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#show 5th image
#plt.imshow(train_images[5], cmap=plt.cm.binary)
#plt.show()


#shrink data information from 0 - 1 
train_images = train_images/255.0
test_images = test_images/255.0


#creates the neural network and define layers for the model

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)), #flattening the data to pass to neural network
	#28*28 = 784 neurons for input layer
	#if information is 2D or 3D, need to flatten it to pass to each individual neuron 
	keras.layers.Dense(128, activation="relu"), #dense layer means fully connected layer, relu = rectified linear unit for activation function
	keras.layers.Dense(10, activation="softmax") #output layer, softmax = probability of the network thinking it is a certain item. between 0 - 1
	])

#model compile configurations
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#epochs = how many times the model will see the same image
#method to increase the model accuracy
model.fit(train_images, train_labels, epochs=3)

#test_loss, test_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction" + class_names[np.argmax(prediction[i])])
	plt.show()