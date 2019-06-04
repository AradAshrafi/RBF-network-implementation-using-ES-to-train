from utils.file_operation import FileOperator

if __name__ == '__main__':
    file_operator = FileOperator()
    csv_file = file_operator.read_from_file()
    print(csv_file)
