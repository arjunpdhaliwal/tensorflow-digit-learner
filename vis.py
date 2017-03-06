import numpy as np
import data
import tensorflow as tf


def model(mInput):
	hiddenw = tf.Variable(tf.random_normal([784, 100]))
	hiddenb = tf.Variable(tf.random_normal([100]))
	
	outputw = tf.Variable(tf.random_normal([100, 10]))
	outputb = tf.Variable(tf.random_normal([10]))

	hl = tf.nn.sigmoid(tf.add(tf.matmul(mInput, hiddenw), hiddenb))

	ol = tf.add(tf.matmul(hl, outputw), outputb)
	return ol;



def runModel(trIn, trOut, teIn, teOut, epochs, learnRate):
	modelInput = tf.placeholder("float", [None, 784])
	modelOutput = tf.placeholder("float")
	network = model(modelInput);

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=modelOutput, logits=network) )
	train = tf.train.GradientDescentOptimizer(learnRate).minimize(cost)
	predict = tf.argmax(network, 1)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			for batchStart in range(0, len(trainingInput), 500):
				batchEnd = batchStart + 500
				sess.run(train, feed_dict={modelInput: trIn[batchStart:batchEnd], modelOutput: trOut[batchStart:batchEnd]})

			print(str(epoch) + ': training data: ' + str(np.mean(np.argmax(trOut, axis=1) == sess.run(predict, feed_dict={modelInput: trIn}))))
			print(str(epoch) + ': test data: ' + str(np.mean(np.argmax(teOut, axis=1) == sess.run(predict, feed_dict={modelInput: teIn}))))


trainingInput, trainingOutput, testInput, testOutput = data.load_formatted_data()
runModel(trainingInput, trainingOutput, testInput, testOutput, 35, 12)