import numpy as np
import random
import math


class RBF:
    # inputs are in X outputs (Labels)are in Y
    # parameters are explained in README
    def __init__(self, RBF_input, RBF_output, centers, gama):
        self.X = RBF_input  # vector of inputs (1D, 2D , ... nD)
        self.Y = RBF_output  # vector of corresponding outputs (labels)
        self.V = centers  # vector of centers, which has same size as inputs
        self.gama = gama  # vector of gama which is a scalar for each V
        self.W = np.asarray([random.random() for _ in range(len(self.V))],
                            dtype=float)  # vectors of weight between RBF and final output(y_prime)
        self.G = np.zeros((len(self.X), len(self.V)))  # matrix of RBF outputs
        self.Y_prime = []  # calculated output
        self.Loss = 0  # Error of last step

    def calculate_G_matrix(self):
        for j in range(len(self.X)):
            for i in range(len(self.V)):
                Gji = math.exp(-self.gama[i] * (np.transpose((self.X[j] - self.V[i])) * (self.X[j] - self.V[i])))
                self.G[j][i] = Gji

    # Y' =  GW
    def calculate_output(self):
        self.Y_prime = self.G * self.W

    # L =  1/2 * (transpose(Y' - Y)) * (Y' - Y)
    def calculate_error(self):
        self.Loss = 1 / 2 * (np.transpose(np.subtract(self.Y_prime, self.Y)) * (np.subtract(self.Y_prime, self.Y)))
