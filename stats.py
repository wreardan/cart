# coding=utf-8
__author__ = 'wesley'


def mean(x):
    return float(sum(x)) / len(x)


def median(x):
    midpoint_index = len(x) // 2
    midpoint = sorted(x)[midpoint_index]
    return midpoint


def min_index(l):
    minimum = min(l)
    for i, val in enumerate(l):
        if val == minimum:
            return i


def max_index(l):
    minimum = max(l)
    for i, val in enumerate(l):
        if val == minimum:
            return i


def mode(lst):
    """http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list"""
    return max(set(lst), key=lst.count)


def basic_linear_regression(x, y):
    """
    solve y = mx + b for m, b given a list of x, y values
    source: http://jmduke.com/posts/basic-linear-regressions-in-python/
    :param x: list of x-values for regression
    :param y: list of y-values for regression
    :return: (m, b)
    """
    #assert(len(x) == len(y))
    length = len(x)
    sum_x = sum(x)
    sum_y = sum(y)

    # Σx^2, and Σxy respectively
    sum_x_squared = sum(map(lambda a: a*a, x))
    covariance = sum([x[i] * y[i] for i in range(length)])

    # Formula
    # B = covariance(x, y) / variance(x)
    numerator = (covariance - (sum_x * sum_y) / length)
    denominator = (sum_x_squared - ((sum_x*sum_x) / length))
    if denominator == 0.0:
        m = 0.0
    else:
        m = numerator / denominator
    b = (sum_y - m * sum_x) / length
    return m, b


def sum_of_squares(x, y, m, b):
    result = 0.0
    for i in range(len(x)):
        expected = m*x[i] + b
        actual = y[i]
        difference = expected - actual
        squared = difference * difference
        result += squared
    return result


def regression_score(matrix, column):
    x = matrix.column(column)
    y = matrix.column(-1)
    m, b = basic_linear_regression(x, y)
    error = sum_of_squares(x, y, m, b)
    return error


def test_regression():
    x = [0, 1, 2]
    y = [3, 5, 8]
    m, b = basic_linear_regression(x, y)
    print(m, b)
    print(sum_of_squares(x, y, m, b))