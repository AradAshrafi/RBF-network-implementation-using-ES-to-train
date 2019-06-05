from utils.file_operation import FileOperator

train_ratio = 0.8
if __name__ == '__main__':
    file_operator = FileOperator()
    csv_file = file_operator
    train_data_input, train_data_output, test_data_input, test_data_output = file_operator.get_test_train_data(
        train_ratio=train_ratio)
