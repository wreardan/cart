#include "tree_node.hpp"
#include "stats.hpp"
#include "util.hpp"

#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

#include <cstdio>

double MINIMUM_GAIN = 0.001;

TreeNode::TreeNode() {
	left = right = NULL;
	column = -1;
	value = 1337.1337;
	gain = -1337.0;
}

TreeNode::~TreeNode() {
	if(left != NULL) {
		delete left;
	}
	if(right != NULL) {
		delete right;
	}
}

void TreeNode::dump(string indent) {

	string next_indent = indent + indent[0];
	if(left != NULL) {

	}
	if(right != NULL) {

	}
}

int TreeNode::count() {
	int result = 1;
	if(left != NULL) {
		result += left->count();
	}
	if(right != NULL) {
		result += right->count();
	}
	return result;
}

double regression_score(Matrix & matrix, int col_index) {
	vector<double> x = matrix.column(col_index);
	vector<double> y = matrix.column(-1);
	double m, b;
	basic_linear_regression(x, y, m, b);
	double error = sum_of_squares(x, y, m, b);
	return error;
}

void TreeNode::train(Matrix & m) {
	cout << "tree training" << endl;
	vector<int> columns = range(m.columns()-1);
	train(m, columns);
}

//Utility function to split X,Y lists
//Returns y values lesser=Y[X<value], greater=Y[X>=value]
void TreeNode::split_xy(const vector<double> & X, const vector<double> & Y, double value, vector<double> & lesser, vector<double> & greater) {
	lesser.empty(); greater.empty();  // make sure output lists are empty
	assert(X.size() == Y.size());  // make sure X,Y are the same size
	//Iterate through elements and put into the lesser/greater output list
	for(int i = 0; i < X.size(); i++) {
		if(X[i] < value) {
			lesser.push_back(Y[i]);
		}
		else {
			greater.push_back(Y[i]);
		}
	}
}
/*
//Gain = parent impurity - sum(child impurity)
double gini_gain(vector<double> parent_classes, vector<double> child1_classes, vector<double> child2_classes, int n_classes) {
	double children_impurity = gini_impurity(child1_classes, n_classes);
	children_impurity += gini_impurity(child2_classes, n_classes);
	return gini_impurity(parent_classes, n_classes) - children_impurity;
}
*/
//Get the feature split values
//Sort by value, then take the midpoint of locations where class switches
vector<double> feature_splits(vector<double> & X, vector<double> & Y) {
	//zip up X and Y
	vector<pair<double, double> > zipped;
	zip(X, Y, zipped);
	//sort
	sort(zipped.begin(), zipped.end(), pairCompare<double>);
	//find class changes
	vector<double> split_values;
	for(int i = 0; i < zipped.size()-1; i++) {
		double class1 = zipped[i].second;
		double class2 = zipped[i+1].second;
		if(class1 != class2) {
			//store midpoint
			double value1 = zipped[i].first;
			double value2 = zipped[i+1].first;
			double midpoint = (value1 + value2) / 2;
			split_values.push_back(midpoint);
		}
	}
	return split_values;
}

//Train the decision tree using Gini Impurity
void TreeNode::train_gini(Matrix & matrix, vector<int> columns, int n_columns, int n_classes) {
	//Test for empty matrix
	assert(matrix.rows() > 0);
	assert(matrix.columns() > 0);

	vector<double> Y = matrix.column(-1);  // class values
	//Check for empty column set
	if(columns.size() == 0) {
		distribution = discrete_p_values(Y);
		return;
	}

	//Select a random subset of columns
	random_shuffle(columns.begin(), columns.end());
	columns.resize(n_columns);

	//Decide which column to split on
	double max_gain = -1000000.0;
	int max_col = 0;
	double max_value = 0.0;
	//For each column(feature)
	for(int i = 0; i < columns.size(); i++) {
		int column_index = columns[i];
		//printf("scanning column %d\n", column_index);
		vector<double> X = matrix.column(column_index);
		//For each value that the class changes
		vector<double> split_values = feature_splits(X, Y);
		for(int j = 0; j < split_values.size(); j++) {
			double v = split_values[j];
			vector<double> lesser, greater;
			split_xy(X, Y, v, lesser, greater);
			double value_gain = gini_gain(Y, lesser, greater, n_classes);
			//printf("value_gain[%d]: %f\n", column_index, value_gain);
			if(value_gain > max_gain) {
				max_gain = value_gain;
				max_value = v;
				max_col = column_index;
			}
		}
	}
	//printf("gain: %f, column: %d, value: %f, rows: %d\n", max_gain, max_col, max_value, matrix.rows());

	//Check for minimum gain
	if(max_gain < MINIMUM_GAIN) {
		//printf("gain less than minimum gain\n");
		distribution = discrete_p_values(Y);
		return;
	}

	//Save split values
	this->column = max_col;
	this->value = max_value;
	this->gain = max_gain;

	//Split datasets
	Matrix l, r;
	matrix.split(max_col, max_value, l, r);

	//Create children
	left = new TreeNode();
	left->train_gini(l, columns, n_columns, n_classes);
	right = new TreeNode();
	right->train_gini(r, columns, n_columns, n_classes);
}

//Train the decision tree using Linear Regression
void TreeNode::train(Matrix & m, vector<int> columns) {
	// pass thru to gini gain training
	train_gini(m, columns);
	return;

	//cout << "training on " << join(columns, ' ') << endl;
	//Edge cases:
	assert(m.rows() > 0);
	assert(m.columns() > 0);
	if(columns.size() == 0) {
		//cout << "column size 0" << endl;
		distribution = discrete_p_values(m.column(-1));
		return;
	}
	//Decide which column to split on
	double min_error = 1000000000.0;
	int min_index = columns[0];
	double error = min_error;
	for(int i = 0; i < columns.size(); i++) {
		int column = columns[i];
		error = regression_score(m, column);
		if(error < min_error) {
			min_index = column;
			min_error = error;
		}
	}
	//Split on lowest error-column
	double v = mean(m.column(min_index));
	Matrix l, r;
	m.split(min_index, v, l, r);
	if(l.rows() <= 0 || r.rows() <= 0) {
		//cout << "l or r: 0 rows" << endl;
		distribution = discrete_p_values(m.column(-1));
		return;
	}
	//cout << l.rows() << ", " << r.rows() << endl;
	double left_error = regression_score(l, min_index);
	double right_error = regression_score(r, min_index);
	double gain = error - (left_error - right_error);
	if(gain < MINIMUM_GAIN) {
		//cout << "split on min gain: " << left_error << " " << right_error << " " << gain << endl;
		distribution = discrete_p_values(m.column(-1));
		return;
	}
	column = min_index;
	value = v;
	//train child nodes in tree
	vector<int> new_columns = columns;
	remove(new_columns.begin(), new_columns.end(), min_index);
	left = new TreeNode();
	left->train(l, new_columns);
	right = new TreeNode();
	right->train(r, new_columns);
	//cout << "Splitton on column " << min_index << " with value " << value << endl;
}

vector<double> TreeNode::classify(vector<double> & row) {
	if(distribution.size() > 0) {
		return distribution;
	}
	if(row[column] < value) {
		assert(left != NULL);
		return left->classify(row);
	}
	else {
		assert(right != NULL);
		return right->classify(row);
	}
}
