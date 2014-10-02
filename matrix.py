from copy import copy
from random import shuffle

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

    def columns(self):
        return len(self.elements[0])

    def column(self, index):
        return [row[index] for row in self.elements]

    def save(self, filename):
        """Save matrix to tab-seperated file"""
        with open(filename, 'w') as f:
            # Write Header
            header = '\t'.join(self.column_labels) + '\n'
            f.write(header)
            # Write elements
            for i, row in enumerate(self.elements):
                string = self.row_labels[i]
                string += '\t'.join(map(float, row))
                f.write(string)

    def load(self, filename, datatype=float, col_headers=True, row_headers=True):
        """
        load a matrix from a tab-seperated file
        :param filename: string path of file to load
        :param col_headers: true if first row is column_labels
        :param row_headers: true if first element in row is the label
        :param datatype: type to cast values to, i.e. int, float, etc.
        :param col_headers: boolean. True if there are column labels on first row
        :return: None
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
        """flatten the matrix into a 1d array"""
        pass

    def dimensions(self):
        return len(self.elements), len(self.elements[0])

    def submatrix(self, rows, columns):
        """
        return a submatrix (does not modify matrix). order DOES matter
        :param rows: list of rows to be included in sub-matrix
        :param columns: list of columns to be included in sub-matrix
        :return:
        """
        s = Matrix(len(rows), len(columns))
        s.elements = [[self.elements[i][j] for j in columns] for i in rows]
        return s

    def transpose(self):
        rows, cols = self.dimensions()
        result = Matrix(cols, rows)
        result.elements = [[self.elements[i][j] for i in range(rows)] for j in range(cols)]
        return result

    def random_split(self):
        N = len(self)
        rows = list(range(len(self.elements)))
        cols = list(range(self.columns()))
        shuffle(rows)
        a = rows[:N//2]
        b = rows[N/2:]
        A = self.submatrix(a, cols)
        B = self.submatrix(b, cols)
        assert(len(A) + len(B) == N)
        return A, B

    def split(self, column, value):
        left = Matrix()
        right = Matrix()
        for row in self.elements:
            if row[column] < value:
                left.elements.append(copy(row))
            else:
                right.elements.append(copy(row))
        return left, right


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
    print()
    print(m.transpose().elements)