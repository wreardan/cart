from copy import copy
from random import shuffle, randint

__author__ = 'wesley'


class Matrix():
    """Data Structure for 2-dimensional data.  You can load and
    save this data structure to files.  You can also slice this
    data in various ways using .submatrix()"""
    def __init__(self, rows=0, columns=0, default=0.0):
        self.elements = [[default for _ in range(columns)] for _ in range(rows)]
        self.row_labels = []
        self.column_labels = []

    def __getitem__(self, item):
        return self.elements[item]

    def __len__(self):
        return len(self.elements)

    def rows(self):
        return len(self)

    def columns(self):
        """Return the number of columns in the matrix"""
        return len(self.elements[0])

    def column(self, index):
        """Get an array of column values at index.  Immutable."""
        return [row[index] for row in self.elements]

    def column_by_name(self, name):
        """Get an array of column values from the column_label.  Immutable."""
        index = self.column_labels.index(name)
        return self.column(index)

    def save(self, filename):
        """Save matrix to tab-separated file"""
        with open(filename, 'w') as f:
            # Write Header
            header = '\t'.join(self.column_labels) + '\n'
            f.write(header)
            # Write elements
            for i, row in enumerate(self.elements):
                string = self.row_labels[i] + '\t'
                string += '\t'.join(map(str, row)) + '\n'
                f.write(string)

    def load(self, filename, datatype=float, col_headers=True, row_headers=True):
        """
        load a matrix from a tab-separated file
        :param filename: string path of file to load
        :param col_headers: true if first row is column_labels
        :param row_headers: true if first element in row is the label
        :param datatype: type to cast values to, i.e. int, float, etc.
        :return: None
        Mutable
        """
        self.elements = []
        with open(filename) as f:
            for i, line in enumerate(f):
                tokens = line.strip().split('\t')
                if len(tokens) == 0:  # Handle blank lines
                    continue
                if col_headers and i == 0:
                    self.column_labels = tokens
                else:
                    if row_headers:
                        self.row_labels.append(tokens[0])
                        tokens = tokens[1:]
                    row = [datatype(value) for value in tokens]
                    self.elements.append(row)

    def flatten(self):
        """flatten the matrix into a 1d array.  Immutable."""
        return [element for row in self.elements for element in row]

    def dimensions(self):
        """return a tuple of (rows, columns)"""
        return len(self.elements), len(self.elements[0])

    def submatrix(self, rows, columns):
        """
        return a submatrix. order DOES matter
        :param rows: list of rows to be included in sub-matrix
        :param columns: list of columns to be included in sub-matrix
        :return: a new matrix with specified rows, cols
        Immutable
        """
        s = Matrix(len(rows), len(columns))
        s.elements = [[self.elements[i][j] for j in columns] for i in rows]
        if len(self.row_labels) > 0:
            s.row_labels = [self.row_labels[i] for i in rows]
        if len(self.column_labels) > 0:
            s.column_labels = [self.column_labels[j] for j in columns]
        return s

    def transpose(self):
        """exchange rows and columns.
        Immutable."""
        rows, cols = self.dimensions()
        result = Matrix(cols, rows)
        result.elements = [[self.elements[i][j] for i in range(rows)] for j in range(cols)]
        result.column_labels = self.row_labels
        result.row_labels = self.column_labels
        return result

    def random_split(self):
        """randomly split the matrix elements into two matrices of
        equal size.
        Immutable"""
        num_rows = len(self)
        rows = list(range(len(self.elements)))
        all_cols = list(range(self.columns()))
        shuffle(rows)
        a_rows = rows[:num_rows//2]
        b_rows = rows[num_rows/2:]
        a = self.submatrix(a_rows, all_cols)
        b = self.submatrix(b_rows, all_cols)
        assert(len(a) + len(b) == num_rows)
        return a, b

    def split(self, column, value):
        """split the matrices into two submatrices based on
        a column's value.
        Immutable"""
        left = Matrix()
        right = Matrix()
        for row in self.elements:
            if row[column] < value:
                left.elements.append(copy(row))
            else:
                right.elements.append(copy(row))
        return left, right

    def get_row(self, label):
        """Get a row based on its label"""
        row_index = self.row_labels.index(label)
        assert(row_index != -1)
        return self.elements[row_index]

    def merge(self, other):
        """Merge another matrix with this Matrix,
        based on row_labels
        Assumes use of row_labels in both matrices
        Mutable"""
        # Append columns
        for label in other.column_labels[1:]:
            self.column_labels.append(label)
        # Add elements
        for label in other.row_labels:
            row = self.get_row(label)
            other_row = other.get_row(label)
            for element in other_row:
                row.append(element)

    def merge_vertical(self, other):
        """Merge matrix with other vertically.
        Mutable"""
        # Add row_labels
        for label in other.row_labels:
            self.row_labels.append(label)
        # Add rows
        for row in other.elements:
            self.elements.append(copy(row))

    def sorted(self, column_index):
        """Returns this matrix sorted on a column.
        Immutable"""
        column = self.column(column_index)
        indices = range(self.rows())
        zipped = list(zip(column, indices))
        zipped.sort()
        rows = [x[1] for x in zipped]
        cols = range(self.columns())
        return self.submatrix(rows, cols)

    def sorted_row_labels(self):
        """Returns this matrix sorted on row labels.
        Immutable"""
        indices = range(self.rows())
        zipped = list(zip(self.row_labels, indices))
        zipped.sort()
        rows = [x[1] for x in zipped]
        cols = range(self.columns())
        return self.submatrix(rows, cols)

    def shuffled(self):
        """Returns a row-shuffled version of this matrix.
        Immutable"""
        rows = range(self.rows())
        shuffle(rows)
        cols = range(self.columns())
        return self.submatrix(rows, cols)

    def random_subset(self, n):
        """returns a random subset of rows in this matrix.
        Immutable.
        TODO: Combine with shuffled() Deduplicate"""
        rows = range(self.rows())
        shuffle(rows)
        rows = rows[:n]
        cols = range(self.columns())
        return self.submatrix(rows, cols)


def test_matrix():
    m = Matrix(4, 3, 0.0)
    m[0][2] = 1.0
    m[1][2] = 2.0
    m[2][2] = 3.0
    m[3][2] = 4.0
    print(m.elements)
    print(m.elements[3])
    print(m.column(2))
    s = m.submatrix(range(4), [2])
    print(s.elements)
    print(m.transpose().elements)
    print(m.flatten())


def test_sort():
    # Test Sort on column
    m = Matrix(3, 2, 0.0)
    m[0][0] = 3.0
    m[1][0] = 2.0
    m[2][0] = 1.0
    m[0][1] = 1.0
    m[1][1] = 2.0
    m[2][1] = 3.0
    r = m.sorted(0)
    assert(r[0][1] == 3.0)
    m.row_labels = ['c', 'b', 'a']
    s = m.sorted_row_labels()
    assert(s[0][1] == 3.0)


if __name__ == '__main__':
    test_matrix()
    test_sort()
