#!/usr/bin/env python
"""Quick missing/incomplete data exercise with NumPy and Pandas."""

import matplotlib.pyplot as plt

import numpy

from pandas import read_csv

import seaborn as sb

dataset = read_csv('pima-indians-diabetes.data.csv', header=None)

# Show the shape (rows & columns) of the dataset
print("Rows, columns = " + str(dataset.shape))
print

# Show the first 20 rows
print("The first 20 observations")
print("-------------------------")
print(dataset.head(20))
print

# count the number of NaN values in each column
print("Number of missing fields (original)")
print("-----------------------------------")
print(dataset.isnull().sum())
print

# Show the stats of the dataset
print("Statistics (original)")
print("---------------------")
print(dataset.describe())
print

fixed_dataset = dataset.copy()

# mark zero values as NaN (missing)
fixed_dataset[[1,2,3,4,5]] = fixed_dataset[[1,2,3,4,5]].replace(0, numpy.NaN)

print("Number of missing fields (zero fields flagged as NaN)")
print("-----------------------------------------------------")
print(fixed_dataset.isnull().sum())
print

# Show the stats of the dataset
print("Statistics (pre-fill)")
print("---------------------")
print(fixed_dataset[[1,2,3,4,5]].describe())
print

# fill missing values with mean column values
fixed_dataset.fillna(value=dataset.mean(), inplace=True)

# count the number of NaN values in each column
print("Number of missing fields (post-fill)")
print("------------------------------------")
print(fixed_dataset.isnull().sum())
print

# Show the stats of the dataset
print("Statistics (post-fill)")
print("----------------------")
print(fixed_dataset[[1,2,3,4,5]].describe())

for i in range(1, 6):
    sb.distplot(dataset[[i]], hist=False)
    fig = sb.distplot(fixed_dataset[[i]], hist=False)
    plt.suptitle('Column ' + str(i))
    plt.show()
