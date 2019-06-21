from .data_location import TWO_CLASSES_PATH, MULTI_CLASSES_PATH, REGRESSION_PATH
import pandas
import random
import numpy as np


class FileOperator:
    def __init__(self, mode="regression"):
        self.path = ""
        # select path based on our mode
        if mode == "regression":
            self.path = REGRESSION_PATH
        elif mode == "two_class":
            self.path = TWO_CLASSES_PATH
        else:
            self.path = MULTI_CLASSES_PATH

    # read from specified CSV file
    def __read_from_file(self):
        try:
            df = pandas.read_csv(self.path)
            return df.values

        except:
            try:
                df = pandas.read_excel(self.path)
                return df.values
            except:
                raise

    # determine test and train data from read CSV file
    def get_test_train_data(self, train_ratio):
        data = self.__read_from_file()
        data_len = len(data)
        test_data = np.copy(data)
        np.random.shuffle(data)
        X_train, Y_train = separate_input_output(data[:int(data_len * train_ratio)])
        X_test, Y_test = separate_input_output(test_data)
        return X_train, Y_train, X_test, Y_test


# consider last column of input as Label and separate X and Y with that assumption
def separate_input_output(labeled_data):
    X = []
    Y = []
    for current_data in labeled_data:
        X.append(current_data[:len(current_data) - 1])
        Y.append(current_data[len(current_data) - 1])
    return X, Y
