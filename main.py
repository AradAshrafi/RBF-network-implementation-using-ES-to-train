from utils.file_operation import FileOperator
from RBF import RBF
from ES import ES
import numpy as np

train_ratio = 0.8
# MU is our population
MU = 10
# I always choose LAMBDA > 2*MU  (7*MU have been proven to be the best portion practically)
# ------ if you want to decrease the LAMBDA to be less than 2*MU beware to apply desired changes in ES codes ------ #
LAMBDA = 7 * MU

if __name__ == '__main__':
    # to read from csv file and put into an array
    file_operator = FileOperator()
    csv_file = file_operator
    train_data_input, train_data_output, test_data_input, test_data_output = file_operator.get_test_train_data(
        train_ratio=train_ratio)

    # cast them to numpy array
    RBF_input = np.asarray(train_data_input)
    RBF_output = np.asarray(train_data_output)

    trained_model = ES(train_data_input=train_data_input, train_data_output=train_data_output, MU=MU, LAMBDA=LAMBDA)
