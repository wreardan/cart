"""
Classification and Regression Trees
Wesley Reardan 2014
"""
import sys
from random import *
from math import *
from copy import *
from itertools import *
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


def list_to_discrete_rv(array, n_classes=2):
    """
    takes a list of positive ints and turns it into
    a discrete distribution
    i.e. list_to_discrete_rv([0,0,1,1,2]) => [0.4, 0.4, 0.2]
    """
    counts = [0] * n_classes
    for element in array:
        counts[element] += 1
    return [float(count)/len(array) for count in counts]


def average_distributions(distributions):
    result = []
    for p_values in zip(*distributions):
        p = float(sum(p_values)) / len(distributions)
        result.append(p)
    return result


def gini_impurity(array):
    # Return 0 impurity for the empty set
    if len(array) == 0:
        return 0.0
    # Get probabilities of element values in array
    probabilities = list_to_discrete_rv(array, 2)
    # Calculate impurity = 1 - sum(squared_probability)
    return 1 - sum([p*p for p in probabilities])


def gini_gain(array, splits):
    # Average child gini impurity
    splits_impurity = sum([gini_impurity(split)*float(len(split))/len(array) for split in splits])
    return gini_impurity(array) - splits_impurity


def weighted_gini_impurity(array, weights):
    if len(array) == 0:
        return 0.0
    probabilities = list_to_discrete_rv(array)
    # Calculate impurity = 1 - sum(squared_probability)
    return 1 - sum([(w*p)*(w*p) for p, w in zip(probabilities, weights)])


CLASS_WEIGHTS = [1,2]
def weighted_gini_gain(array, splits, weights=CLASS_WEIGHTS):
    # Average child gini impurity
    splits_impurity = sum([weighted_gini_impurity(split,weights)*float(len(split))/len(array) for split in splits])
    return weighted_gini_impurity(array,weights) - splits_impurity


def entropy(array):
    # Return 0 entropy for the empty set
    if len(array) == 0:
        return 0.0
    # Get probabilities of element values in array
    probabilities = list_to_discrete_rv(array)
    # Sum -p_i * log2(p_i)
    return sum([-p * log(p,2) for p in probabilities if p > 0.0])

def information_gain(array, splits):
    # Average child entropy
    splits_entropy = sum([entropy(split)*float(len(split))/len(array) for split in splits])
    return entropy(array) - splits_entropy


def max_p(array):
    probabilities = list_to_discrete_rv(array)
    return max(probabilities)

def max_p_gain(array, splits):
    "https://homes.cs.washington.edu/~etzioni/papers/brute-aai94.pdf"
    split_max_p = max([max_p(split) for split in splits])
    p = max_p(array)
    return split_max_p - p


def normalize(array):
    s = sum(array)
    for i in range(len(array)):
        array[i] /= s

def normalize_square(array):
    pass


GAIN_FUNCTION = gini_gain
MINIMUM_GAIN = 0.01
MINIMUM_NUM_SAMPLES = 40
PRUNE_OOB = 1.0
BALANCED_CLASSIFIER = True
WEIGHTED_CLASSIFIER = False
N_FEATURES = 7
N_TREES = 50
P_SAMPLES = 1.0


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
        """Returns a subset of the matrix
        based on row_indices - a list of row index"""
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
        """Splits Matrix into two based on a value in column.
        First Matrix is column<value
        Second Matrix is column>=value
        """
        lesser = Matrix()
        greater = Matrix()
        for row in self:
            if row[column] < value:
                lesser.append(copy(row))
            else:
                greater.append(copy(row))
        return lesser, greater

    def sample_class_rows(self, column, class_value, n_samples):
        """Samples with replacement from class.
        Returns row indices"""
        rows = [i for i in range(len(self)) if self[i][column] == class_value]
        return [choice(rows) for _ in range(n_samples)]


class Node():
    """
    The heart of the algorithm: the decision tree.
    Can use various gain measures: gini, information_gain, weighted_gini, max_p
    Caution: recursion in heavy use :)
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.distribution = None
        self.split_value = None
        self.split_column = None
        self.gain = None

    def size(self):
        """Returns the size of the tree"""
        if self.left and self.right:
            return 1 + self.left.size() + self.right.size()
        return 1

    def dump(self, indent=' '):
        """Dump the Tree to a string.
        Indent should be any single character.
        """
        if self.left and self.right:
            assert(len(indent) > 0)
            print(indent + 'split node [%d]<%f gain: %f' % (self.split_column, self.split_value, self.gain))
            self.left.dump(indent + indent[0])
            self.right.dump(indent + indent[0])
        else:
            distr = ' '.join(map(str, self.distribution.distribution))
            print(indent + 'leaf node: ' + distr)

    def split(self, X, Y, splitval):
        """utility function to split X,Y lists by value"""
        lesser = [y for x,y in zip(X, Y) if x < splitval]
        greater_equal = [y for x,y in zip(X, Y) if x >= splitval]
        return lesser, greater_equal

    def feature_splits(self, X, Y):
        """Calculate possible split-values.
        Sorts the X,Y values by X.
        Next, finds points where class changes and use the median
        between those two points
        """
        split_values = []
        sorted_xy = sorted(zip(X, Y))
        for i in range(1, len(sorted_xy)):
            if sorted_xy[i-1][1] != sorted_xy[i][1]:
                average = (sorted_xy[i-1][0] + sorted_xy[i][0]) / 2.0
                split_values.append(average)
        return split_values

    def best_split(self, matrix, column_list, n_features):
        """Find the best split based on variables in column_list
        use number of features specified."""
        subset = sample(column_list, n_features)
        y = matrix.column(-1)
        max_gain = -1000000.0
        max_col = None
        max_val = None
        for column in subset:
            x = matrix.column(column)
            split_values = self.feature_splits(x, y)
            for splitval in split_values:
                splits = self.split(x, y, splitval)
                #print(len(splits[0]), len(splits[1]), len(x), splitval)
                gain = GAIN_FUNCTION(y, splits)
                if gain > max_gain:
                    max_col = column
                    max_val = splitval
                    max_gain = gain
        #if max_col is not None: print('best feature [%d]<%f with gain %f, len(%d)' % (max_col, max_val, max_gain, len(matrix)))
        #print(best_rv.lower, best_rv.upper, best_rv.delta, max_val)
        #assert(max_val < best_rv.upper)
        return max_col, max_val, max_gain

    def save_class_distribution(self, matrix):
        """Get the class distribution for the Matrix and save
        as this node (leaf node)"""
        self.distribution = list_to_discrete_rv(matrix.column(-1))
        # Apply weights
        if WEIGHTED_CLASSIFIER:
            for i in range(len(self.distribution)):
                self.distribution[i] *= CLASS_WEIGHTS[i]
        assert(len(self.distribution) > 0)

    def train(self, matrix, column_list, n_features=7, parent_gain=0.0):
        """
        Train the Classification Tree.
        column_list: list of columns to train data
        n_features: number of features to split on
        parent_gain: dont touch this parameter (for recursive use depth>1)
            its used make sure the tree doesn't get stuck in gain loops
        """
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
            # Train Recursively
            self.left = Node()
            self.left.train(left_matrix, column_list, n_features, gain)
            self.right = Node()
            self.right.train(right_matrix, column_list, n_features, gain)
        except AssertionError:
            self.save_class_distribution(matrix)

    def classify(self, row):
        """classify sample of data row.
        Returns the class distribution.
        """
        if self.left and self.right:
            if row[self.split_column] < self.split_value:
                return self.left.classify(row)
            else:
                return self.right.classify(row)
        return self.distribution


class Forest():
    """
    Basic Random Forest Classifier

    Training is done on all data

    Classification is done with an unweighted average of the
    tree distributions
    """
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


def max_index(x):
    """O(2N)"""
    minimum = max(x)
    for i, val in enumerate(x):
        if val == minimum:
            return i


def parallel_train(state):
    """Child process training function"""
    # Unpack parameters
    matrix, columns, n_features, p_samples = state
    n_samples = int(len(matrix) * p_samples)
    oob_error = PRUNE_OOB
    cycles = 0
    MAX_CYCLES = 10
    while oob_error >= PRUNE_OOB and cycles < MAX_CYCLES:
        # Generate subsets
        row_indices = list(range(len(matrix)))
        if BALANCED_CLASSIFIER:
            training_rows = []
            num_classes = len(set(matrix.column(-1)))
            for cls in range(num_classes):
                rows = matrix.sample_class_rows(-1, cls, int(n_samples/num_classes))
                training_rows.extend(rows)
        else:
            training_rows = [choice(row_indices) for _ in range(n_samples)]
        training = matrix.rows(training_rows)
        testing_rows = [r for r in row_indices if r not in training_rows]
        testing = matrix.rows(testing_rows)
        # Create and train tree
        root = Node()
        root.train(training, columns, n_features)
        # Calculate out-of-bag (oob) error
        right = 0
        wrong = 0
        for row in testing:
            dist = root.classify(row)
            c = max_index(dist)
            if c == row[-1]:
                right += 1
            else:
                wrong += 1
        oob_error = float(wrong) / (right + wrong)
        cycles += 1
    root.oob_error = oob_error
    return root


class ParallelForest(Forest):
    """Parallel extension of Random Forest algorithm
    Creates child processes to train trees inside a Process Pool
    Classification is done in serial from inherited method
    """
    def __init__(self, n_trees=100, n_features=7, processes=0, p_samples=P_SAMPLES):
        self.n_trees = n_trees
        self.trees = []
        self.n_features = n_features
        self.p_samples = p_samples
        if processes <= 0:
            processes = cpu_count() - 1
        self.pool = Pool(processes)

    def train(self, matrix, features):
        star = [(matrix, features, self.n_features, self.p_samples) for _ in range(self.n_trees)]
        self.trees = self.pool.map(parallel_train, star)
        for tree in self.trees:
            # tree.dump()
            print('oob error(%d): %f' % (tree.size(), tree.oob_error))


def cross_fold_validation(matrix, classifier, args, n_folds=10):
    """
    Perform n-fold validation using classifier on matrix of data
    Passes args into the classifier when calling __init__
    """
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
                print(result, set(matrix.column(-1)))
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
    forest_type = 'parallel'
    if forest_type == 'parallel':
        parallel_forest_args = (N_TREES, N_FEATURES, 4)
        aupr = cross_fold_validation(m, ParallelForest, parallel_forest_args)
    if forest_type == 'regular':
        forest_args = (N_TREES, N_FEATURES)
        aupr = cross_fold_validation(m, Forest, forest_args)
    with open(sys.argv[2], 'w') as f:
        for p, cls in aupr:
            f.write('%f\t%d\n' % (p, cls))


if __name__ == '__main__':
    main()
