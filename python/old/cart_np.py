import numpy as np
from pandas import DataFrame, read_csv
from random import shuffle, sample
from numpy import mean, polyfit, dot
from scipy.stats import linregress
from copy import copy
import sys


MINIMUM_GAIN = 0.1


class TreeNode():
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.value = None
        self.distribution = None

    def feature_error(self, dataframe, feature, classes=None):
        if classes is None:
            classes = dataframe['Class']
        values = dataframe[feature]
        coefficients, residuals, _, _, _ = polyfit(values, classes, 1, full=True)
        if len(residuals) <= 0:
            print(len(classes), len(values), feature)
            raise Exception()

        return coefficients, residuals[0]

    def class_distribution(self, classes):
        """
        Set this node's class distribution.
        :param classes: list of class values from matrix
        :return: None
        """
        pass

    def train(self, dataframe, feature_names, n_features):
        # Select subset of features
        feature_subset = sample(feature_names, n_features)
        # Find highest scoring feature
        classes = dataframe['Class']
        min_error = 1000000
        best_feature = feature_subset[0]
        best_ce = None
        for f in feature_subset:
            ce, err = self.feature_error(dataframe, f, classes)
            if err < min_error:
                min_error = err
                best_feature = f
                best_ce = ce
        print('best feature %s with err %f' % (best_feature, min_error))

        # Split
        left_dataframe = dataframe[dataframe[best_feature] * best_ce[0] < best_ce[1]]
        right_dataframe = dataframe[dataframe[best_feature] * best_ce[0] >= best_ce[1]]
        print(len(left_dataframe), len(right_dataframe))
        #TODO: check length > MINIMUM_something
        # Check gain
        _, left_error = self.feature_error(left_dataframe, best_feature)
        _, right_error = self.feature_error(right_dataframe, best_feature)
        gain = min_error-(left_error+right_error)
        print(best_feature, gain)
        if gain > MINIMUM_GAIN:
            self.left = TreeNode()
            self.left.train(left_dataframe, feature_names, n_features)
            self.right = TreeNode()
            self.right.train(right_dataframe, feature_names, n_features)
        else:
            self.class_distribution(classes)

    def classify(self, row):
        pass


def main():
    # Load data matrix with pandas
    filename = sys.argv[1]
    frame = read_csv(filename, sep='\t', header=0, index_col=0)
    #print(frame['Class'])
    all_column_names = list(frame.columns.values)
    data_column_names = all_column_names[:-1]

    tree = TreeNode()
    tree.train(frame, data_column_names, 7)


if __name__ == '__main__':
    main()