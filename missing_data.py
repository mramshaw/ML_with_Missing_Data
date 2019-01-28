#!/usr/bin/env python

# pylint: disable=C0103,C0325,W0621

"""Quick missing/incomplete data exercise with NumPy and Pandas."""

import matplotlib.pyplot as plt

import numpy

from pandas import read_csv

import seaborn as sb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def k_fold_cross_validation(dataset, title):
    """Define the common code for k-fold cross-validation."""
    # split dataset into inputs and outputs
    values = dataset.values
    X = values[:, 0:8]
    y = values[:, 8]

    # evaluate an LDA model on the dataset using k-fold cross validation
    model = LinearDiscriminantAnalysis()
    kfold = KFold(n_splits=3, random_state=7)
    result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(title)
    underline = ""
    for _ in xrange(len(title)):
        underline += "-"
    print(underline)
    print(result.mean())


if __name__ == '__main__':

    dataset = read_csv('pima-indians-diabetes.data.csv', header=None)

    # Show the shape (rows & columns) of the dataset
    print("Rows, columns = " + str(dataset.shape))
    print

    # Show the first 20 rows
    print("The first 20 observations")
    print("-------------------------")
    print(dataset.head(20))
    print

    # count the number of zero values - where zero is an anomaly
    print("Number of zero values")
    print("---------------------")
    print((dataset[[1, 2, 3, 4, 5, 6, 7]] == 0).sum())
    print

    # count the number of NaN values (using isnull) in each column
    print("Number of missing fields (original)")
    print("-----------------------------------")
    print(dataset.isnull().sum())
    print

    # Show the stats of the dataset
    print("Statistics (original)")
    print("---------------------")
    print(dataset.describe())
    print

    # Make a copy of the dataset so we can compare original & replaced
    replaced_dataset = dataset.copy()

    # mark zero values as NaN (missing)
    replaced_dataset[[1, 2, 3, 4, 5]] = \
        replaced_dataset[[1, 2, 3, 4, 5]].replace(0, numpy.NaN)

    print("Number of missing fields (zero fields flagged as NaN)")
    print("-----------------------------------------------------")
    print(replaced_dataset.isnull().sum())
    print

    # Show the stats of the dataset
    print("Statistics (pre-fill)")
    print("---------------------")
    print(replaced_dataset[[1, 2, 3, 4, 5]].describe())
    print

    # Make copies of the dataset so we can compare them
    mean_dataset = replaced_dataset.copy()
    mode_dataset = replaced_dataset.copy()
    median_dataset = replaced_dataset.copy()

    # fill missing values with mean column values
    mean_dataset.fillna(value=replaced_dataset.mean(), inplace=True)

    # count the number of NaN values in each column
    print("Number of missing fields (post-fill)")
    print("------------------------------------")
    print(mean_dataset.isnull().sum())
    print

    # Show the stats of the dataset
    print("Statistics (post-fill)")
    print("----------------------")
    print(mean_dataset[[1, 2, 3, 4, 5]].describe())
    print

    # fill missing values with column mode value
    mode_dataset.fillna(value=replaced_dataset.mode(numeric_only=True).iloc[0],
                        inplace=True)

    # fill missing values with column median value
    median_dataset.fillna(value=replaced_dataset.median(), inplace=True)

    # Use Seaborn to plot before & after graphs for columns 1 - 5
    for i in range(1, 6):
        sb.distplot(dataset[[i]], hist=False, label='Original')
        # Cannot plot datasets with NaN values
        sb.distplot(mode_dataset[[i]], hist=False, label='Mode')
        sb.distplot(median_dataset[[i]], hist=False, label='Median')
        sb.distplot(mean_dataset[[i]], hist=False, label='Mean')
        plt.suptitle('Column ' + str(i))
        plt.show()

    # split dataset into inputs and outputs
    values = replaced_dataset.values
    X = values[:, 0:8]
    y = values[:, 8]

    # evaluate an LDA model on the dataset using k-fold cross validation
    # [We know it is going to fail, so wrap it in a try/catch block.]
    print("Accuracy (with NaN values)")
    print("--------------------------")
    try:
        model = LinearDiscriminantAnalysis()
        kfold = KFold(n_splits=3, random_state=7)
        result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        print(result.mean())
    except ValueError as ve:
        print
        print ve
    print

    # ------------------------------
    # drop NaN entries and try again
    # ------------------------------

    # create new dataset with NaN entries removed
    dropped_dataset = replaced_dataset.dropna(inplace=False)

    # summarize the number of rows and columns in the dataset
    print("Rows, columns (NaN values dropped) = " + str(dropped_dataset.shape))
    print

    # Show the stats of the dataset
    print("Statistics (NaN values dropped)")
    print(dropped_dataset[[1, 2, 3, 4, 5]].describe())
    print

    k_fold_cross_validation(dropped_dataset,
                            "Accuracy (with NaN values dropped)")
    print

    # -------------------------------------
    # use mean-filled entries and try again
    # -------------------------------------

    k_fold_cross_validation(mean_dataset, "Accuracy (with NaN values filled)")
    print

    # Use Seaborn to plot before & after graphs for columns 1 - 5
    for i in range(1, 6):
        sb.distplot(dataset[[i]], hist=False, label='Original')
        sb.distplot(dropped_dataset[[i]], hist=False, label='NaN Dropped')
        sb.distplot(mode_dataset[[i]], hist=False, label='NaN Filled (mean)')
        plt.suptitle('Column ' + str(i))
        plt.show()
