from utils.file_operation import FileOperator
from ES import ES
import numpy as np

train_ratio = 0.8
# MU is our population
MU = 35
# I always choose LAMBDA > 2*MU  (7*MU have been proven to be the best portion practically)
LAMBDA = 7 * MU

# 3 modes had been evaluated
# 1-regression, 2- classification(two-classes) 3-classification(multi-classes)
# ----------------------------------------- important --------------------------->
# COMMANDS ARE ::::::: 1-"regression", 2-"two_class",   3-"multi_class"
mode = "multi_class"

# number of different centers in each RBF(number of V)
number_of_centers = 6

# total iterations in ES
total_iterations = 100

if __name__ == '__main__':
    # to read from csv o excel file and put into an array
    file_operator = FileOperator(mode=mode)
    csv_file = file_operator
    train_data_input, train_data_output, test_data_input, test_data_output = file_operator.get_test_train_data(
        train_ratio=train_ratio)

    # cast them to numpy array
    RBF_input = np.array(train_data_input)
    RBF_output = np.array(train_data_output)

    # Perform ES to train our model, then test it with test dataset
    trained_model = ES(train_data_input=train_data_input, train_data_output=train_data_output, MU=MU, LAMBDA=LAMBDA,
                       number_of_centers=number_of_centers, total_iterations=total_iterations, mode=mode)
    trained_model.test_accuracy(test_data_input=test_data_input, test_data_output=test_data_output, mode=mode)
