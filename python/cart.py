import sys
from copy import copy
from random import shuffle, sample, choice
from multiprocessing.pool import Pool
from itertools import chain

from matrix import Matrix
from stats import mean, regression_score, list_to_discrete, add_distributions,\
    max_index, counts_to_probabilities


MINIMUM_GAIN = 0.1


def sample_with_replacement(l, k):
    return [choice(l) for _ in range(k)]


class TreeNode():
    """A class to represent a Node in the Regression Tree"""
    def __init__(self):
        self.left = self.right = None
        self.column = None
        self.value = None
        self.probabilities = None

    def train(self, matrix, columns):
        """Train the regression tree on a matrix of data using features
        in columns"""
        assert(len(matrix) > 0)
        if len(columns) <= 0:
            self.probabilities = list_to_discrete(matrix.column(-1))
            return
        # Decide which column to split on
        min_error = 1000000000
        min_index = columns[0]
        error = min_error
        for col_index in columns:
            error = regression_score(matrix, col_index)
            #print(col_index, error)
            if error < min_error:
                min_index = col_index
                min_error = error
        # Split on lowest-error column
        value = mean(matrix.column(min_index))
        left, right = matrix.split(min_index, value)
        if len(left) <= 0 or len(right) <= 0:
            self.probabilities = list_to_discrete(matrix.column(-1))
            return
        left_error = regression_score(left, min_index)
        right_error = regression_score(right, min_index)
        gain = error-(left_error+right_error)
        # Stop recursing if below threshhold
        if gain < MINIMUM_GAIN:
            self.probabilities = list_to_discrete(matrix.column(-1))
            return
        #print(gain, min_index, min_error)
        # Set self values
        self.column = min_index
        self.value = value
        # Create new child nodes
        new_columns = copy(columns)
        new_columns.remove(min_index)
        self.left = TreeNode()
        self.left.train(left, new_columns)
        self.right = TreeNode()
        self.right.train(right, new_columns)

    def classify(self, row):
        """Classify a vector of data using this regression tree"""
        if self.probabilities is not None:
            return self.probabilities
        if row[self.column] < self.value:
            return self.left.classify(row)
        else:
            return self.right.classify(row)


class Forest():
    """A forest contains a collection of trees with partial
    feature sets.  These trees vote on a classification
    to determine the final class."""
    def __init__(self, n_trees=100, n_features=10, p_samples=0.1):
        self.n_trees = n_trees
        self.n_features = n_features
        self.p_samples = p_samples
        self.trees = []
        for i in range(n_trees):
            tree = TreeNode()
            self.trees.append(tree)

    def train(self, matrix):
        all_columns = list(range(matrix.columns()-1))
        n_samples = int(self.p_samples * len(matrix))
        for tree in self.trees:
            shuffle(all_columns)
            columns = all_columns[0:self.n_features]
            subset = matrix.random_subset(n_samples)
            tree.train(subset, columns)

    def classify(self, row):
        distributions = []
        for tree in self.trees:
            dist = tree.classify(row)
            #dist = counts_to_probabilities(dist)
            distributions.append(dist)
        total_counts = add_distributions(distributions)
        #print(total_counts, row[-1])
        p = float(total_counts[1])/sum(total_counts)
        return max_index(total_counts), p


class BalancedRandomForest(Forest):
    def train(self, matrix):
        n_samples = int(self.p_samples * len(matrix))
        all_columns = list(range(matrix.columns()))
        data_columns = all_columns[:-1]
        matrices = matrix.discrete_split(-1)
        assert(len(matrices) > 1)
        for tree in self.trees:
            # Select subset of features
            columns = copy(data_columns)
            shuffle(columns)
            columns = columns[0:self.n_features]
            shuffle(all_rows_zero)
            # Sample with or without replacement from classes
            training_set = Matrix()
            for m in matrices:
                row_indices = list(range(len(m)))
                shuffle(row_indices)
                row_indices = row_indices[0:n_samples/2]
                training_set.merge_vertical(m.submatrix(row_indices, all_columns))
            #print(subzero.column(-1))
            tree.train(training_set, columns)


def parallel_train(state):
    """function to train trees in another process
    state is (matrix, columns) because we cannot use
    a starmap in Python < 3.3 """
    matrix, columns, n_samples = state
    #m = Matrix()
    #m.load(matrix_filename)
    m = matrix.random_subset(n_samples)
    root = TreeNode()
    root.train(m, columns)
    return root


def select_subset(array, number):
    """returns a random subset of array of size number.
    Immutable."""
    col_set = copy(array)
    shuffle(col_set)
    return col_set[:number]


class ParallelForest(Forest):
    """A parallel implementation of Random Forest.
    It starts a Pool of processes, then uses map to create a
    set of trees.  Classification is still done in serial:
    inherits Forest's classify method."""
    def __init__(self, n_trees=100, n_features=10, p_samples=0.1, n_processes=2):
        self.n_trees = n_trees
        self.n_features = n_features
        self.p_samples = p_samples
        self.pool = Pool(n_processes)
        self.trees = []

    def train(self, matrix):
        all_columns = list(range(matrix.columns()-1))
        n_samples = int(self.p_samples * len(matrix))
        star = [(matrix, select_subset(all_columns, self.n_features), n_samples) for _ in range(self.n_trees)]
        self.trees = self.pool.map(parallel_train, star)


def evaluate(matrix, classifier):
    right = wrong = 0
    p_values = []
    classes = []
    for i in range(len(matrix)):
        row_class, p = classifier.classify(matrix[i])
        p_values.append(p)
        classes.append(row_class)
        expected_class = matrix[i][-1]
        if row_class == expected_class:
            right += 1
        else:
            wrong += 1
    return right, wrong, p_values, classes


def cross_fold_validation(matrix, classifier, arguments=(), n_folds=10):
    R = len(matrix)
    N = R / n_folds
    all_columns = range(0, matrix.columns())
    total_percent = 0.0
    all_p_values = []
    all_classes = []
    for fold in range(n_folds):
        cls = classifier(*arguments)
        if fold == 0:  # Beginning Fold
            training_rows = range(N, R)
        elif fold < n_folds-1:  # Middle Folds
            training_rows = chain(range(0, fold*N), range((fold+1)*N, R))
        else:  # End Fold
            training_rows = range(0, R-N)
        testing_rows = range(fold*N, (fold+1)*N)
        if fold == n_folds-1:
            testing_rows = range(fold*N, R)
        #print(len(matrix), matrix.columns())
        testing = matrix.submatrix(list(testing_rows), all_columns)
        training = matrix.submatrix(list(training_rows), all_columns)
        cls.train(training)
        right, wrong, p_values, classes = evaluate(testing, cls)
        all_p_values.extend(p_values)
        all_classes.extend(classes)
        percent = right * 100.0 / len(testing)
        print("fold %d: %f%%" % (fold, percent))
        total_percent += percent
    total_percent /= n_folds
    return total_percent, all_p_values, all_classes


def main():
    if len(sys.argv) < 2:
        print('usage: python cart.py training_file')
        exit(1)
    # Load Matrices
    train = Matrix()
    train.load(sys.argv[1])
    matrix = train.shuffled()  # Shuffle Matrix so classes are spread out
    # args = (500, 7, 0.1)  # num_trees, num_features, num_samples
    # percent, p_values, all_classes = cross_fold_validation(matrix, Forest, args)
    # fargs = (500, 7, 0.1, 4)
    # percent, p_values, all_classes = cross_fold_validation(matrix, ParallelForest, fargs)
    args = (500, 7, 0.1)
    percent, p_values, all_classes = cross_fold_validation(matrix, BalancedRandomForest, args)
    print("total percent: %f%%" % percent)

    # Write soft labels
    actual_classes = matrix.column(-1)
    with open('./data/p_values.txt', 'w') as f:
        f.write('Gene Name\tP Value\tWes Class\tActual Class\n')
        for name, p, class_id, actual in zip(matrix.row_labels, p_values, all_classes, actual_classes):
            f.write('%s\t%f\t%f\t%f\n' % (name, p, class_id, actual))



if __name__ == '__main__':
    main()
