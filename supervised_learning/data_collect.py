import csv

import numpy as np


def data_write1():
    csv_filepath_train ='./train_data_label.csv'
    csv_address = 'train_1.csv'
    writer_train = csv.writer(open(csv_filepath_train, 'w'))
    writer_train.writerow(['address'])
    writer_train.writerow([csv_address])
def data_write2():
    csv_filepath_train = './train_data_label.csv'
    rule_csv_address = 'train_1.csv'
    limit_csv_address = 'limit_1.csv'
    writer_train = csv.writer(open(csv_filepath_train, 'w'))
    writer_train.writerow(['rule_address','limit_address'])
    writer_train.writerow([rule_csv_address, limit_csv_address])
#data_write2()
def data_write_limit(limit,grid_size):

    csv_filepath_train = './data_cls/limit_6.csv'

    writer_train = csv.writer(open(csv_filepath_train, 'w'))
    writer_train.writerow(['limit', 'grid_size'])
    writer_train.writerow([limit,grid_size])



limit = np.zeros((5,4))

limit[0] = np.array([36, 45, 42, 54])

limit[1] = np.array([26, 42, 42, 54])

limit[2] = np.array([15, 36, 15, 36])

limit[3] = np.array([15, 36, 15, 36])

limit[4] = np.array([9, 45, 9, 120])

limit = np.array2string(limit, precision=2, separator=',',

                      suppress_small=True)

grid_size = np.array([45,120])

grid_size = np.array2string(grid_size, precision=2, separator=',',

                      suppress_small=True)

data_write_limit(limit,grid_size)

