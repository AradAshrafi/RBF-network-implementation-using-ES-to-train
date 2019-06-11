import numpy as np
import random
import math


class RBF:
    # inputs are in X outputs (Labels)are in Y
    # parameters are explained in README
    def __init__(self, RBF_input, RBF_output, centers_number=10, V=None, GAMA=None):
        self.X = RBF_input  # vector of inputs (1D, 2D , ... nD)
        self.Y = RBF_output  # vector of corresponding outputs (labels)

        # first we initialize it by giving random value
        # vector of centers, which has same size as inputs
        self.V = V if (V is not None) else [random.random() * RBF_input[int(random.random() * len(RBF_input))] for _ in
                                            range(centers_number)]

        # vector of gama which is a scalar for each V
        self.GAMA = GAMA if (GAMA is not None) else np.asarray([random.uniform(0, 0.01) for _ in
                                                                range(centers_number)])
        self.W = []  # vectors of weight between RBF and final output(y_prime)
        # np.asarray([random.random() for _ in range(len(self.V))])
        self.G = np.zeros((len(self.X), len(self.V)))  # matrix of RBF outputs
        self.Y_prime = []  # calculated output
        self.Loss = 0  # Error of last step

    def set_input(self, RBF_input):
        self.X = RBF_input

    def set_output(self, RBF_output):
        self.X = RBF_output

    # it returns chromosome of current RBF :)
    # our chromosomes are V and GAMA
    def get_chromosome(self):
        return [self.GAMA, self.V]

    def set_chromosome(self, new_gama, new_centers):
        self.GAMA = new_gama
        self.V = new_centers

    # calculate G matrix based on formula written in README.md file
    def calculate_G_matrix(self):
        for j in range(len(self.X)):
            for i in range(len(self.V)):
                Gji = math.exp(-self.GAMA[i] * (np.dot((self.X[j] - self.V[i]).T, (self.X[j] - self.V[i]))))
                self.G[j][i] = Gji

    # Y' =  GW
    def calculate_output(self):
        self.Y_prime = np.dot(self.G, self.W)

    # L =  1/2 * (transpose(Y' - Y)) * (Y' - Y)
    def calculate_error(self):
        self.Loss = np.dot((self.Y_prime - self.Y).T, (self.Y_prime - self.Y))

    # Calculate Weights Matrix  based on formula written
    def calculate_weights(self):
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(self.G.T, self.G)), self.G.T), self.Y)

