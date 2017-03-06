import cPickle
import gzip

import numpy as np

#adapted from the loader presented at http://neuralnetworksanddeeplearning.com/chap1.html - i've made some changes to fit what I'm doing

def load_data():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, test_data)

def load_formatted_data():
    tr_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    test_inputs = [np.reshape(x, (784)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (training_inputs, training_results, test_inputs, test_results)

def vectorized_result(j):
    e = np.zeros((10))
    e[j] = 1.0
    return e