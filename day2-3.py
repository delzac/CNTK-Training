import csv
import numpy as np

data_filepath = "C:/Users/delza/OneDrive/Work/SAFORO Internal Training/Deep learning with CNTK/datasets/creditcard.csv"

with open(data_filepath, 'r') as f:
    reader = csv.reader(f)

    next(reader)  # removes header
    data_x = []
    data_y = []
    for line in reader:
        data_x.append(line[:-1])
        data_y.append(line[-1])

data_x = np.array(data_x)  # converts list into np.array
print(data_x)
print(data_x.shape)
print(data_x.dtype)
