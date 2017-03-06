import numpy as np
import data
import tensorflow as tf


def model(mInput):
	hiddenw = tf.Variable(tf.random_normal([784, 100]))  
	hiddenb = tf.Variable(tf.random_normal([100])) #randomly initialize weights and biases for the hidden layer
	
	outputw = tf.Variable(tf.random_normal([100, 10]))
	outputb = tf.Variable(tf.random_normal([10])) #randomly initialize weights and biases for the output layer

	hl = tf.nn.sigmoid(tf.add(tf.matmul(mInput, hiddenw), hiddenb)) #we'll use the sigmoid function for our neuron model

	ol = tf.add(tf.matmul(hl, outputw), outputb)
	return ol;



def runModel(trIn, trOut, teIn, teOut, epochs, learnRate):
	modelInput = tf.placeholder("float", [None, 784]) #we want 784-dimensional images as our input
	modelOutput = tf.placeholder("float")
	network = model(modelInput); #create a model based on our input specification 

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=modelOutput, logits=network)) #the cost function
	train = tf.train.GradientDescentOptimizer(learnRate).minimize(cost) #so  we'll use stochastic gradient descent as our optimizer 
	predict = tf.argmax(network, 1) #we can verify predictions by picking the largest number in each vector 

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer()) #initialize all variables

		for epoch in range(epochs): 
			for batchStart in range(0, len(trainingInput), 500): #train in batches of 500
				batchEnd = batchStart + 500
				sess.run(train, feed_dict={modelInput: trIn[batchStart:batchEnd], modelOutput: trOut[batchStart:batchEnd]})

			print(str(epoch) + ': training data: ' + str(np.mean(np.argmax(trOut, axis=1) == sess.run(predict, feed_dict={modelInput: trIn}))))
			print(str(epoch) + ': test data: ' + str(np.mean(np.argmax(teOut, axis=1) == sess.run(predict, feed_dict={modelInput: teIn})))) #print training results and compare with test data in each epoch


trainingInput, trainingOutput, testInput, testOutput = data.load_formatted_data() 
runModel(trainingInput, trainingOutput, testInput, testOutput, 35, 12) #load the data and run the model!
