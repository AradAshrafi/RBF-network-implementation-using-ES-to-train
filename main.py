from utils.file_operation import FileOperator
from ES import ES
import numpy as np

train_ratio = 0.8
# MU is our population
MU = 10
# I always choose LAMBDA > 2*MU  (7*MU have been proven to be the best portion practically)
# ------ if you want to decrease the LAMBDA to be less than 2*MU beware to apply desired changes in ES codes ------ #
LAMBDA = 3 * MU

# 3 modes had been evaluated
# 1-regression, 2- classification(two-classes) 3-classification(multi-classes)
# COMMANDS ARE ::::::: 1-"regression", 2-"two-class",   3-"multi-class"
mode = "multi_classes"

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
    trained_model = ES(train_data_input=train_data_input, train_data_output=train_data_output, MU=MU, LAMBDA=LAMBDA)
    trained_model.test_accuracy(test_data_input=test_data_input, test_data_output=test_data_output, mode=mode)
