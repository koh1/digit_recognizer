import numpy as np
import pylab
import pandas as pd
import sys

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - x ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1-x)

def identity(x):
    return x

def identity_deriv(x):
    return 1

class MultiLayerPerceptron:
    def __init__(self, num_input, num_hidden, num_output, act1="tanh", act2="sigmoid"):
        if act1 == "tanh":
            self.act1 = tanh
            self.act1_deriv = tanh_deriv
        elif act1 == "sigmoid":
            self.act1 = sigmoid
            self.act1_deriv = sigmoid_deriv
        else:
            print "ERROR: act1 is tanh or sigmoid"
            sys.exit()

        if act2 == "tanh":
            self.act2 = tanh
            self.act2_deriv = tanh_deriv
        elif act2 == "sigmoid":
            self.act2 = sigmoid
            self.act2_deriv = sigmoid_deriv
        elif act2 == "identity":
            self.act2 = identity
            self.act2_deriv = identity_deriv
        else:
            print "ERROR: act1 is tanh or sigmoid or identity"
            sys.exit()

        self.num_input = num_input + 1
        self.num_hidden = num_hidden + 1
        self.num_output = num_output
        
        self.weight1 = np.random.uniform(-1.0, 1.0, (self.num_hidden, self.num_input))
        self.weight2 = np.random.uniform(-1.0, 1.0, (self.num_output, self.num_hidden))


    def fit(self, X, t, learning_rate=0.01, epochs=10000):
        X = np.hstack([np.ones([X.shape[0],1]),X])
        t = np.array(t)
        
        for k in range(epochs):
            print k
            
            i = np.random.randint(X.shape[0])
            x = X[i]
            
            z = self.act1(np.dot(self.weight1, x))
            y = self.act2(np.dot(self.weight2, z))
            
#            delta2 = self.act2_deriv(y) * (y - t[i])
            delta2 = y - t[i]
            delta1 = self.act1_deriv(z) * np.dot(self.weight2.T, delta2)

            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)
            self.weight1 -= learning_rate * np.dot(delta1.T, x)

            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)
            self.weight2 -= learning_rate * np.dot(delta2.T, z)

    def predict(self, x):
        x = np.array(x)

        x = np.insert(x, 0, 1)
        z = self.act1(np.dot(self.weight1, x))
        y = self.act2(np.dot(self.weight2, z))
        
        return y

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(2, 2, 1, "tanh", "sigmoid")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    t = np.array([0,1,1,0])
    mlp.fit(X,t)
    for i in [[0,0],[0,1],[1,0],[1,1]]:
        print i, mlp.predict(i)

