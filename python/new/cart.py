"""
Classification and Regression Trees
Wesley Reardan 2014
"""
import sys
from random import *
from math import *
from rv import *
from copy import *
from itertools import *
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

def gini_impurity(array):
    # Get probabilities of element values in array
    probabilities = DiscreteRandomVariable(array).distribution
    # Calculate impurity = 1 - sum(squared_probability)
    return 1 - sum([p*p for p in probabilities])


def entropy(array):
    # Get probabilities of element values in array
    probabilities = DiscreteRandomVariable(array).distribution
    # Sum -p_i * log2(p_i)
    return sum([-p * log(p,2) for p in probabilities if p > 0.0])


def information_gain(array, splits):
    # Average child entropy
    splits_entropy = sum([entropy(split) for split in splits]) / len(splits)
    return entropy(array) - splits_entropy


MINIMUM_GAIN = 0.01
MINIMUM_NUM_SAMPLES = 40


class Matrix(list):
    """
    Generic Matrix class
    2d datastructure represented as a list of lists
    """
    def load(self, filename, delimiter='\t'):
        """
        Load Matrix from filename.
        Trys to parse elements:
        int, float, then string.
        """
        del self[:]
        with open(filename, 'r') as f:
            for line in f:
                row = []
                elements = line.strip().split(delimiter)
                for element in elements:
                    try:
                        element = int(element)
                    except ValueError:
                        try:
                            element = float(element)
                        except ValueError:
                            pass  # keep as string
                    row.append(element)
                self.append(row)

    def column(self, index):
        """
        Returns a single column inside the Matrix.
        """
        return [row[index] for row in self]

    def rows(self, row_indices):
        m = Matrix()
        for index in row_indices:
            m.append(copy(self[index]))
        return m

    def save(self, filename):
        """
        Save the Matrix to a file.
        """
        with open(filename, 'w') as f:
            for row in self:
                f.write('\t'.join(map(str,row)) + '\n')

    def split_on_value(self, column, value):
        lesser = Matrix()
        greater = Matrix()
        for row in self:
            if row[column] < value:
                lesser.append(copy(row))
            else:
                greater.append(copy(row))
        return lesser, greater


class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.distribution = None
        self.split_value = None
        self.split_column = None
        self.gain = None

    def size(self):
        if self.left and self.right:
            return 1 + self.left.size() + self.right.size()
        return 1

    def dump(self, indent=' '):
        if self.left and self.right:
            print(indent + 'split node [%d]<%f gain: %f' % (self.split_column, self.split_value, self.gain))
            self.left.dump(indent + indent[0])
            self.right.dump(indent + indent[0])
        else:
            distr = ' '.join(map(str, self.distribution.distribution))
            print(indent + 'leaf node: ' + distr)

    def split(self, X, Y, splitval):
        lesser = [y for x,y in zip(X, Y) if x < splitval]
        greater_equal = [y for x,y in zip(X, Y) if x >= splitval]
        return lesser, greater_equal

    def best_split(self, matrix, column_list, n_features):
        subset = sample(column_list, n_features)
        y = matrix.column(-1)
        max_gain = -1000000.0
        max_col = None
        max_val = None
        for column in subset:
            x = matrix.column(column)
            rv = ContinuousRandomVariable(x, 1000)
            for _ in range(100):
                splitval = rv.sample()
                splits = self.split(x, y, splitval)
                #print(len(splits[0]), len(splits[1]), len(x), splitval)
                gain = information_gain(y, splits)
                if gain > max_gain:
                    max_col = column
                    max_val = splitval
                    max_gain = gain
                    best_rv = rv
        #print('best feature [%d]<%f with gain %f, len(%d)' % (max_col, max_val, max_gain, len(matrix)))
        #print(best_rv.lower, best_rv.upper, best_rv.delta, max_val)
        assert(max_val < best_rv.upper)
        return max_col, max_val, max_gain

    def save_class_distribution(self, matrix):
        self.distribution = DiscreteRandomVariable(matrix.column(-1))
        assert(len(self.distribution.distribution) > 0)

    def train(self, matrix, column_list, n_features=7, parent_gain=0.0):
        # Check for stopping criteria
        try:
            assert(len(column_list) > n_features)
            assert(len(matrix) > MINIMUM_NUM_SAMPLES)
            # Find Best Split
            col, value, gain = self.best_split(matrix, column_list, n_features)
            assert(gain > MINIMUM_GAIN)
            assert(gain != parent_gain)
            # Save split values
            self.split_column = col
            self.split_value = value
            self.gain = gain
            # Split datasets
            left_matrix, right_matrix = matrix.split_on_value(col, value)
            #print('before assertions %d %d', (len(left_matrix), len(right_matrix)))
            #print(len(left_matrix), len(right_matrix))
            assert(len(left_matrix) > 0)
            assert(len(right_matrix) > 0)
            # Remove column from list and pass down the chain
            new_column_list = list(column_list)
            new_column_list.remove(col)
            # Train Recursively
            self.left = Node()
            self.left.train(left_matrix, new_column_list, n_features, gain)
            self.right = Node()
            self.right.train(right_matrix, new_column_list, n_features, gain)
        except AssertionError:
            self.save_class_distribution(matrix)

    def classify(self, row):
        if self.left and self.right:
            if row[self.split_column] < self.split_value:
                return self.left.classify(row)
            else:
                return self.right.classify(row)
        return self.distribution.distribution

class Forest():
    def __init__(self, n_trees=100, n_features=7):
        self.trees = []
        for _ in range(n_trees):
            self.trees.append(Node())
        self.n_features = n_features

    def train(self, matrix, features):
        for tree in self.trees:
            tree.train(matrix, features, self.n_features)

    def classify(self, row):
        distributions = []
        for tree in self.trees:
            dist = tree.classify(row)
            distributions.append(dist)
        avg = average_distributions(distributions)
        return avg


def parallel_train(state):
    matrix, columns, n_features = state
    root = Node()
    root.train(matrix, columns, n_features)
    return root


class ParallelForest(Forest):
    def __init__(self, n_trees=100, n_features=7, processes=0):
        self.n_trees = n_trees
        self.trees = []
        self.n_features = n_features
        if processes <= 0:
            processes = cpu_count() - 1
        self.pool = Pool(processes)

    def train(self, matrix, features):
        star = [(matrix, features, self.n_features) for _ in range(self.n_trees)]
        self.trees = self.pool.map(parallel_train, star)


def cross_fold_validation(matrix, classifier, args, n_folds=10):
    # Shuffle Matrix
    shuffle(matrix)
    # Train and Test in Folds
    S = int(len(matrix) / n_folds)
    p_values = []
    classes = []
    for fold in range(n_folds):
        # Separate Data into train/test sets
        testing_rows = range(fold*S,(fold+1)*S)
        if fold == n_folds-1:
            testing_rows = chain(testing_rows, range(n_folds*S, len(matrix)))
        testing_matrix = matrix.rows(testing_rows)
        training_rows = range(0, S*fold)
        if fold != n_folds-1:
            training_rows = chain(training_rows, range((fold+1)*S, len(matrix)))
        training_matrix = matrix.rows(training_rows)
        # Train Classifier
        cls = classifier(*args)
        features = range(1, len(matrix[0])-1)
        cls.train(training_matrix, features)
        # Validate testing set
        for row in testing_matrix:
            result = cls.classify(row)
            if len(result) < 2:
                p = 0.0
            else:
                p = result[1]
            p_values.append(p)
            classes.append(row[-1])
        print('fold %d completed' % fold)
    return zip(p_values, classes)

def main():
    m = Matrix()
    m.load(sys.argv[1])
    del(m[0])  # Delete Header row
    # tree = Node()
    # tree.train(m, list(range(1,45)), 7)
    # tree.dump()
    #print('%d nodes in tree' % tree.size())
    # forest = Forest(1, 7)
    # forest.train(m, range(1, 45))
    # with open(sys.argv[2], 'w') as f:
    #     for row in m:
    #         dist = forest.classify(row)
    #         if len(dist) < 2:
    #             dist.append(0.0)
    #         f.write('%f\t%d\n' % (dist[1], row[-1]))
    forest_args = (10, 7)
    aupr = cross_fold_validation(m, ParallelForest, forest_args)
    with open(sys.argv[2], 'w') as f:
        for p, cls in aupr:
            f.write('%f\t%d\n' % (p, cls))

if __name__ == '__main__':
    main()
