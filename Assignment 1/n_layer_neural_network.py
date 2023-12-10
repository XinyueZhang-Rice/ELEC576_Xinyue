from three_layer_neural_network import NeuralNetwork
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)

    # Add new datasets make_blobs and make_circles
    # X, y = datasets.make_blobs(n_samples=200, random_state=20)
    # X, y = datasets.make_circles(n_samples=200, factor=.5, noise=.05)
    return X, y



def add_type(fn, type):
    def wrapper(*args, **kwargs):
        return fn(type=type, *args, **kwargs)

    return wrapper


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,  input_dim, hidden_dim, nn_hidden_layers, output_dim, actFun_type='Sigmoid', reg_lambda=0.01, seed=0):
        '''
        :param input_dim: input dimension
        :param hidden_dim: the number of hidden units
        :param nn_hidden_layers: the number of hidden layers
        :param output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nn_hidden_layers = nn_hidden_layers
        self.output_dim = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.seed = seed

        # initialize input layer
        input_layer = Layer(self.input_dim, self.hidden_dim, add_type(self.actFun, actFun_type),
                            add_type(self.diff_actFun, actFun_type))
        self.layers = [input_layer]

        #initialize hidden layers
        for n in range(self.nn_hidden_layers):
            hidden_layer = Layer(hidden_dim, hidden_dim, add_type(self.actFun, actFun_type),
                                 add_type(self.diff_actFun, actFun_type))
            self.layers += [hidden_layer]

        #initialize output layer
        self.W_output = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.b_output = np.zeros((1, self.output_dim))

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.feedforward(X) # use feedforward function implemented in Layer class

        #z = x * w + b
        self.z_output = np.dot(X, self.W_output) + self.b_output
        self.probs = np.exp(self.z_output) / np.sum(np.exp(self.z_output), axis=-1, keepdims=True)


    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)

        # Calculating the loss
        y_stack = np.stack((1 - y, y), -1)

        data_loss = np.sum(-(y_stack * np.log(self.probs)))
        #print(np.concatenate([layer.W.ravel() for layer in self.layers]))
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(np.concatenate([layer.W.ravel() for layer in self.layers]))))
        return (1. / num_examples) * data_loss


    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):

        # calculate the dw for the last layer
        y_stack = np.stack((1 - y, y), -1)
        self.dW_out = np.dot(self.layers[-1].a.T, self.probs - y_stack)
        #print(self.W_output.T.shape)
        #print((self.probs - y_stack).shape)
        da = np.dot(self.probs - y_stack, self.W_output.T)

        for layer in reversed(self.layers):
            layer.backprop(da)
            da = layer.dX



    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        # Gradient descent.
        for i in range(0, num_passes):
            self.feedforward(X)
            self.backprop(X, y)
            self.dW_out += self.reg_lambda * self.W_output


            for layer in self.layers:
                layer.dW += self.reg_lambda * layer.W
            self.W_output += -epsilon * self.dW_out

            for layer in self.layers:
                layer.W += -epsilon * layer.dW
                layer.b += -epsilon * layer.db

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


class Layer():
    def __init__(self, input_dim, output_dim, actFun, diff_actFun, seed=0):
        '''
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param actFun: activation functions computation from three_layer_neural_network.py
        :param seed: random seed
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun = actFun
        self.diff_actFun = diff_actFun

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))

    def feedforward(self, X):
        #z = wx+b
        #a = acFun(z)
        self.X = X # X is the input data
        self.z = np.dot(X, self.W) + self.b # z is the input of the current layer
        self.a = self.actFun(self.z)
        return self.a

    def backprop(self, da):
        '''
        :param da = dL/da
        '''
        # dW = dL/dW
        # db = dL/db
        # dX = dL/dX
        num_examples = len(self.X)
        self.dW = np.dot(self.X.T, da * (self.diff_actFun(self.z)))
        self.db = np.dot(np.ones(num_examples), da * (self.diff_actFun(self.z)))
        self.dX = np.dot((da * self.diff_actFun(self.z)), self.W.T)


if __name__ == "__main__":
    X, y = generate_data()
    model = DeepNeuralNetwork( input_dim=2, hidden_dim=10, nn_hidden_layers=1, output_dim=2, actFun_type="ReLU")
    model.fit_model(X, y, epsilon=0.01)
    model.visualize_decision_boundary(X, y)