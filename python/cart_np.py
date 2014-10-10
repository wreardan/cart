import numpy as np
from numpy import mean
from scipy.stats import mode
from copy import copy
import sys


class Node():
    def __init__(self):
        self.left = self.right = None
        self.column = None
        self.value = None
        self.classification = None
        self.gain = None

    def error(self, matrix, column, y=None):
        if y is None:
            y = matrix[:,-1]
        x = matrix[:,column]
        assert(len(x) == len(y) and len(x) != 0)
        _, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
        if len(residuals) == 0:
            return 1000000000.0
        return residuals[0]

    def train(self, matrix, columns, min_gain=0.1):
        assert(len(matrix) > 0)
        if len(columns) <= 0:
            self.classification = mode(matrix[:,-1])
            return

        # Decide which column to split
        y = matrix[:,-1]
        min_error = 1000000000
        min_index = columns[0]
        for column in columns:
            error = self.error(matrix, column, y)
            if error < min_error:
                min_index = column
                min_error = error
        value = mean(matrix[:,min_index])
        num_cols = matrix.shape[1]
        all_cols = list(range(num_cols))
        left = matrix[np.argwhere(matrix[:,min_index] < value), all_cols]
        print matrix.shape
        right = matrix[np.argwhere(matrix[:,min_index] >= value), all_cols]
        print matrix.shape
        #print(len(left), len(right), len(matrix)) # two extra columns WTF?!
        left_error = self.error(left, min_index)
        right_error = self.error(right, min_index)
        gain = error - (left_error + right_error)
        if gain < min_gain:
            self.classification = mode(matrix[:,-1])
            return
        print(gain, min_index, min_error)
        # Set self values
        self.column = min_index
        self.value = value
        # Create new child nodes
        new_columns = copy(columns)
        new_columns.remove(min_index)
        self.left = Node()
        self.left.train(left, new_columns)
        self.right = Node()
        self.right.train(right, new_columns)


filename = sys.argv[1]
# Get column labels and number of columns
with open(filename) as f:
    col_labels = f.readline().split()
    num_cols = len(col_labels)
col_range = range(1, num_cols) # omit first column

matrix = np.loadtxt(filename, delimiter='\t', skiprows=1, usecols=col_range)
'''
y = matrix[:,-1]
scores = []
for column in range(num_cols-1):
    x = matrix[:,column]
    tupl = np.polyfit(x, y, 2, full=True)
    #print(tupl)
    coeff, residuals, rank, singular_values, threshhold = tupl
    if len(residuals > 0):
        scores.append(residuals[0])
    #print(np.linalg.lstsq(z, y))
scores.sort()
print(scores[-1])
'''

node = Node()
col_range = list(range(num_cols-1))
node.train(matrix, col_range)
