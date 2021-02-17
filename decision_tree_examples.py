import numpy as np
import pandas as pd


if __name__ == "__main__":
    input_file = "data/cereal.csv"

    # easily pull in csv data with pandas dataframes
    df = pd.read_csv(input_file)

    # add the original column names to a python list
    original_headers = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    numpy_array = df.as_matrix()
