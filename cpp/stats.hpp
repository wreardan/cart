#pragma once

#include <vector>
using std::vector;

double sum(const vector<double> & list);
double sum_squared(const vector<double> & list);
double mean(const vector<double> & list);
double mode(const vector<double> & list);
double covariance(const vector<double> & dist1, const vector<double> & dist2);
void basic_linear_regression(const vector<double> & x, const vector<double> & y, double & m, double & b);
double sum_of_squares(const vector<double> & x, const vector<double> & y, double m, double b);
void test_regression();

