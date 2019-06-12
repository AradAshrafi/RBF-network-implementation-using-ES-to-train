import numpy as np
import random
import math
import matplotlib.pyplot as plt


class RBF:
    # inputs are in X outputs (Labels)are in Y
    # parameters are explained in README
    def __init__(self, RBF_input=None, RBF_output=None, centers_number=5, V=None, GAMA=None):
        self.X = RBF_input  # vector of inputs (1D, 2D , ... nD)
        self.Y = RBF_output  # vector of corresponding outputs (labels)

        # first we initialize it by giving random value
        # vector of centers, which has same size as inputs
        self.V = V if (V is not None) else [random.uniform(0.1, 0.8) * RBF_input[int(random.random() * len(RBF_input))]
                                            for _ in
                                            range(centers_number)]

        # vector of gama which is a scalar for each V
        self.GAMA = GAMA if (GAMA is not None) else np.asarray([random.uniform(0, 0.01) for _ in
                                                                range(centers_number)])
        self.W = []  # vectors of weight between RBF and final output(y_prime)
        # np.asarray([random.random() for _ in range(len(self.V))])
        # self.G = np.zeros((len(self.X), len(self.V))) if (self.X is not None) else None  # matrix of RBF outputs
        self.G = None
        self.Y_prime = []  # calculated output
        self.Loss = 0  # Error of last step

    # it returns chromosome of current RBF :)
    # our chromosomes are V and GAMA
    def get_chromosome(self):
        return [self.GAMA, self.V]

    def set_chromosome(self, new_gama, new_centers):
        self.GAMA = new_gama
        self.V = new_centers

    # calculate G matrix based on formula written in README.md file
    def calculate_G_matrix(self):
        self.G = np.zeros((len(self.X), len(self.V)))
        for j in range(len(self.X)):
            for i in range(len(self.V)):
                Gji = math.exp(-self.GAMA[i] * (np.dot((self.X[j] - self.V[i]).T, (self.X[j] - self.V[i]))))
                self.G[j][i] = Gji

    # Y' =  GW
    def calculate_output(self):
        self.Y_prime = np.dot(self.G, self.W)

    # L =  1/2 * (transpose(Y' - Y)) * (Y' - Y)
    def calculate_error(self):
        self.Loss = 1 / 2 * np.dot((self.Y_prime - self.Y).T, (self.Y_prime - self.Y))

    # Calculate Weights Matrix based on formula written
    def calculate_weights(self):
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(self.G.T, self.G)), self.G.T), self.Y)

    # for after initialization and testing time ---------------------------------------------------------------------->
    def set_input(self, RBF_input):
        self.X = RBF_input
        if self.X is None:
            self.W = None
            self.G = None

    def set_output(self, RBF_output):
        self.Y = RBF_output

    def normalize_output(self, number_of_classes=2):
        if number_of_classes == 2:
            for i in range(len(self.Y_prime)):
                self.Y_prime[i] = 1 if self.Y_prime[i] > 0 else -1
        else:
            print()

    # We Have Two Modes Here :
    # 1-linear Regression
    # 2-classification
    def test_accuracy(self, test_data_input, test_data_output, mode="regression"):
        self.set_input(RBF_input=test_data_input)
        self.set_output(RBF_output=test_data_output)
        # first we calculate G matrix based on centers(v) and gama [formula is written in README ]
        self.calculate_G_matrix()
        # then weights
        self.calculate_weights()
        # then output based on G and W( Y_prime = GW)
        self.calculate_output()
        if mode == "regression":
            self.__plot_regression_accuracy()
        else:
            # then we normalize our output
            self.normalize_output()
            self.__plot_classification_accuracy()

    def __plot_regression_accuracy(self):
        plt.plot(self.Y, color="g")
        plt.plot(self.Y_prime, color="b")
        plt.show()

    def __plot_classification_accuracy(self):
        accuracy_counter = 0
        correct_X = []
        correct_Y = []
        incorrect_X = []
        incorrect_Y = []
        center_X = []
        center_Y = []
        for i in range(len(self.V)):
            center_X.append(self.V[i][0])
            center_Y.append(self.V[i][1])

        for i in range(len(self.Y_prime)):
            if self.Y_prime[i] == self.Y[i]:
                accuracy_counter += 1
                correct_X.append(self.X[i][0])
                correct_Y.append(self.X[i][1])
            else:
                incorrect_X.append(self.X[i][0])
                incorrect_Y.append(self.X[i][1])

        plt.plot(correct_X, correct_Y, '.', color='g')
        plt.plot(incorrect_X, incorrect_Y, '.', color='r')
        plt.plot(center_X, center_Y, 'o', color='b')
        plt.show()

        accuracy = accuracy_counter / len(self.Y)
        print(accuracy)
