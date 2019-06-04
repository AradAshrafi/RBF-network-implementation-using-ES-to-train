from .data_location import TWO_CLASSES_PATH
import pandas


class FileOperator:
    def __init__(self, path=TWO_CLASSES_PATH):
        self.path = path

    def read_from_file(self):
        df = pandas.read_csv(self.path)
        return df
