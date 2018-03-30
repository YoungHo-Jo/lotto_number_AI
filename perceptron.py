import numpy as np

class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, inputSize, numberOfNuerons=1, learningRate = 1, numberOfEpochs = 100):
        # Add one for bias
        # Insert the bias into the input when performing the weight update
        self.W = np.array([np.zeros(inputSize + 1) for _ in range(numberOfNuerons)])

        self.numberOfEpochs = numberOfEpochs
        self.learningRate = learningRate
        self.numberOfNuerons = numberOfNuerons
        self.inputSize = inputSize
    
    def activiation(self, x):
        # return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0

    def predict(self, x):
        if x.shape[0] != (self.inputSize + 1):
            x = np.insert(x, 0, 1)
        predicted = np.zeros(self.numberOfNuerons)
        max = (0, -1) 
        for i in range(self.W.shape[0]):
            z = self.W[i].T.dot(x)
            if max[1] == -1 or z > max[0]:
                max = (z, i)
        predicted[max[1]] = self.activiation(max[0])
        return predicted

    def training(self, X, d):
        for _ in range(self.numberOfEpochs):
            for i in range(self.W.shape[0]):
                for j in range(d.shape[0]):
                    x = np.insert(X[j], 0, 1)
                    predicted = self.predict(x)
                    error = d[j] - predicted
                    self.W[i] = self.W[i] + self.learningRate * error[i] * x



# if __name__ == '__main__':
#     X = np.array([
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [1, 1]
#     ])
#     d = np.array([0, 0, 0, 1])

#     perceptron = Perceptron(inputSize = 2)
#     perceptron.training(X, d)
#     print(perceptron.W)


    
