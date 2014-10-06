/*
General Statistics Functions
*/
#include "stats.hpp"
#include <iostream>
#include <map>
#include <algorithm>

using namespace std;

double sum(const vector<double> & list) {
	double result = 0.0;
	for(int i = 0; i < list.size(); i++) {
		result += list[i];
	}
	return result;
}

double sum_squared(const vector<double> & list) {
	double result = 0.0;
	for(int i = 0; i < list.size(); i++) {
		result += list[i] * list[i];
	}
	return result;
}

double mean(const vector<double> & list) {
	return sum(list) / list.size();
}

double mode(const vector<double> & list) {
	map<double, int> repeats;
	for(int i = 0; i < list.size(); i++) {
		double value = list[i];
		if(repeats.find(value) == repeats.end()) {
			//not found
			repeats[value] = 1;
		}
		else {
			repeats[value] += 1;
		}
	}
	//http://stackoverflow.com/questions/9370945/c-help-finding-the-max-value-in-a-map
	map<double, int>::iterator max = max_element(repeats.begin(), repeats.end(), 
		[](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second < p2.second; }
   	);
	return (*max).first;
}

double covariance(const vector<double> & dist1, const vector<double> & dist2) {
	double result = 0.0;
	for(int i = 0; i < dist1.size(); i++) {
		result += dist1[i] * dist2[i];
	}
	return result;
}

void basic_linear_regression(const vector<double> & x, const vector<double> & y, double & m, double & b) {
	int length = x.size();
	double sum_x = sum(x);
	double sum_y = sum(y);

	double sum_x_squared = sum_squared(x);
	double cov = covariance(x, y);

	double numerator = (cov - (sum_x * sum_y) / length);
	double denominator = (sum_x_squared - ((sum_x*sum_x) / length));
	if(denominator == 0.0) {
		m = 0.0;
	}
	else {
		m = numerator / denominator;
	}
	b = (sum_y - m * sum_x) / length;
}

double sum_of_squares(const vector<double> & x, const vector<double> & y, double m, double b) {
	double result = 0.0;
	for(int i = 0; i < x.size(); i++) {
		double expected = m * x[i] + b;
		double actual = y[i];
		double difference = expected - actual;
		double squared = difference * difference;
		result += squared;
	}
	return result;
}

void test_regression() {
	vector<double> x;
	x.push_back(0.0);
	x.push_back(1.0);
	x.push_back(2.0);
	vector<double> y;
	y.push_back(3.0);
	y.push_back(5.0);
	y.push_back(8.0);
	double m, b;
	basic_linear_regression(x, y, m, b);
	cout << "y = " << m << "*x + " << b << endl;
}

