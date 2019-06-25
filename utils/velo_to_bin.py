import numpy as np
from numpy import genfromtxt

import timeit


def convert_data():
    np_data = genfromtxt('../datasets/2018-06-13-13-45-05_Velodyne-VLP-16-Data_Frame_0081.csv',
                         delimiter=",", skip_header=1, usecols=(3, 4, 5, 6), dtype=np.float32)
    np_data.tofile("output/0001.bin")
