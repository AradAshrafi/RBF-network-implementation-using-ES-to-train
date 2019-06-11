from .data_location import TWO_CLASSES_PATH
import pandas


class FileOperator:
    def __init__(self, path=TWO_CLASSES_PATH):
        self.path = path

    # read from specified CSV file
    def __read_from_file(self):
        df = pandas.read_csv(self.path)
        return df.values

    # determine test and train data from read CSV file
    def get_test_train_data(self, train_ratio):
        data = self.__read_from_file()
        data_len = data.__len__()
        X_train, Y_train = separate_input_output(data[:int(data_len * train_ratio)])
        X_test, Y_test = separate_input_output(data[int(train_ratio * data_len):])
        return X_train, Y_train, X_test, Y_test


# consider last column of input as Label and separate X and Y with that assumption
def separate_input_output(labeled_data):
    X = []
    Y = []
    for current_data in labeled_data:
        X.append(current_data[:len(current_data) - 1])
        Y.append(current_data[len(current_data) - 1])
    return X, Y
