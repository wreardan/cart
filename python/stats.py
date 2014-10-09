# coding=utf-8
__author__ = 'wesley'


def mean(x):
    """O(N)"""
    return float(sum(x)) / len(x)


def median(x):
    """O(N*log(N))"""
    midpoint_index = len(x) // 2
    midpoint = sorted(x)[midpoint_index]
    return midpoint


def min_index(x):
    """O(2N)"""
    minimum = min(x)
    for i, val in enumerate(x):
        if val == minimum:
            return i


def max_index(x):
    """O(2N)"""
    minimum = max(x)
    for i, val in enumerate(x):
        if val == minimum:
            return i


def mode(x):
    """
    return the most common element in a list
    source: http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
    """
    return max(set(x), key=x.count)


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
    # TODO: make this into a series of unit tests
    x = [0, 1, 2]
    y = [3, 5, 8]
    m, b = basic_linear_regression(x, y)
    print(m, b)
    print(sum_of_squares(x, y, m, b))


def chi2(distribution1, distribution2, degrees_of_freedom=2):
    """
    X^2 test for two distributions
    source: Tyler Harter, cs776 hw#3, mdd.predict.py:Node.chi_squared()
    :param distribution1: list of counts
    :param distribution2: list of counts
    :return: X^2 score
    """
    sum1 = sum(distribution1)
    sum2 = sum(distribution2)
    total = float(sum1 + sum2)
    score = 0.0
    for count1, count2 in zip(distribution1, distribution2):
        rowsum = count1 + count2
        expected1 = (rowsum / total) * sum1
        expected2 = (rowsum / total) * sum2
        if expected1 != 0:
            score += (count1 - expected1) ** 2 / expected1
        if expected2 != 0:
            score += (count2 - expected2) ** 2 / expected2
    return score


def chi2_score(matrix, column):
    """chi2 score for column against classification"""
    x = matrix.column(column)
    y = matrix.column(-1)
    score = chi2(x, y)
    return score


def chi2_score2(matrix, column):
    """chi2 score for column against other columns"""
    score = 0.0
    x = matrix.column(column)
    for j in range(matrix.columns()-1):
        if j != column:
            y = matrix.column(j)
            score += chi2(x, y)
    return score / ((matrix.columns() - 1) ** 2)