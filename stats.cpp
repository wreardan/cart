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

//http://xoax.net/cpp/ref/cpp_examples/incl/mean_med_mod_array/
double mode(const vector<double> & daArray) {
    // Allocate an int array of the same size to hold the
    // repetition count
    int iSize = daArray.size();
    int* ipRepetition = new int[iSize];
    for (int i = 0; i < iSize; ++i) {
        ipRepetition[i] = 0;
        int j = 0;
        bool bFound = false;
        while ((j < i) && (daArray[i] != daArray[j])) {
            if (daArray[i] != daArray[j]) {
                ++j;
            }
        }
        ++(ipRepetition[j]);
    }
    int iMaxRepeat = 0;
    for (int i = 1; i < iSize; ++i) {
        if (ipRepetition[i] > ipRepetition[iMaxRepeat]) {
            iMaxRepeat = i;
        }
    }
    delete [] ipRepetition;
    return daArray[iMaxRepeat];
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



/*
Convert a list of numbers into a class distribution
*/
vector<int> list_to_discrete(const vector<double> & list, int num_classes) {
	vector<int> classes(num_classes, 0);
	for(int i = 0; i < list.size(); i++) {
		int cls = (int)list[i];
		assert(cls >= 0);
		assert(cls < num_classes);
		classes[cls] += 1;
	}
	return classes;
}

vector<double> discrete_p_values(const vector<double> & list, int num_classes) {
	//Convert list to discrete RV
	vector<int> discrete = list_to_discrete(list, num_classes);
	//Convert to probability values
	vector<double> p_values;
	for(int i = 0; i < num_classes; i++) {
		int count = discrete[i];
		double p = ((double) count) / list.size();
		p_values.push_back(p);
	} 
	return p_values;
}

void add_counts(vector<int> & list1, vector<int> & list2) {
	if(list1.size() == 0) {
		for(int i = 0; i < list2.size(); i++) {
			list1.push_back(0);
		}
	}
	assert(list1.size() == list2.size());
	for(int i = 0; i < list1.size(); i++) {
		list1[i] += list2[i];
	}
}

//Returns the sum of p^2 when list of classes is converted to a Discrete Random Variable
#include <cstdio>
double gini_impurity(const vector<double> & classes, int num_classes) {
	if(classes.size() == 0) {
		return 0.0;
	}
	vector<double> p_values = discrete_p_values(classes, num_classes);
	//printf("p_values[0]=%f, p_values[1]=%f\n", p_values[0], p_values[1]);
	double result = 1.0;
	for(int i = 0; i < p_values.size(); i++) {
		double p = p_values[i];
		result -= p * p;
	}
	return result;
}
double gini_gain(const vector<double> & parent_classes,
				const vector<double> & child1_classes,
				const vector<double> & child2_classes,
				int num_classes) {
	double children_impurity = 0.0;
	children_impurity += gini_impurity(child1_classes, num_classes) * child1_classes.size() / parent_classes.size();
	children_impurity += gini_impurity(child2_classes, num_classes) * child2_classes.size() / parent_classes.size();
	double parent_impurity = gini_impurity(parent_classes, num_classes);
	//printf("parent: %f, children: %f\n", parent_impurity, children_impurity);
	return parent_impurity - children_impurity;
}

//Weighted Gini Impurity
// multiply p by weight inside impurity calculation
double weighted_gini_impurity(const vector<double> & classes,
							const vector<double> & class_weights,
							int num_classes) {
	vector<double> p_values = discrete_p_values(classes, num_classes);
	double result = 1.0;
	for(int i = 0; i < p_values.size(); i++) {
		double p = p_values[i];
		double w = class_weights[i];
		result -= (w*p) * (w*p);
	}
	return result;
}

double weighted_gini_gain(const vector<double> & parent_classes,
				const vector<double> & child1_classes,
				const vector<double> & child2_classes,
				const vector<double> & class_weights,
				int num_classes) {
	double children_impurity = weighted_gini_impurity(child1_classes, class_weights, num_classes)
		+ weighted_gini_impurity(child2_classes, class_weights, num_classes);
	double parent_impurity = weighted_gini_impurity(parent_classes, class_weights, num_classes);
	return parent_impurity - children_impurity;
}