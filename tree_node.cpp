#include "tree_node.hpp"
#include "stats.hpp"
#include "util.hpp"

#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

double MINIMUM_GAIN = 0.1;

TreeNode::TreeNode() {
	left = right = NULL;
	column = -1;
	value = 1337.1337;
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

void TreeNode::train(Matrix & m, vector<int> columns) {
	//cout << "training on " << join(columns, ' ') << endl;
	//Edge cases:
	assert(m.rows() > 0);
	assert(m.columns() > 0);
	if(columns.size() == 0) {
		//cout << "column size 0" << endl;
		class_counts = list_to_discrete(m.column(-1));
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
		class_counts = list_to_discrete(m.column(-1));
		return;
	}
	//cout << l.rows() << ", " << r.rows() << endl;
	double left_error = regression_score(l, min_index);
	double right_error = regression_score(r, min_index);
	double gain = error - (left_error - right_error);
	if(gain < MINIMUM_GAIN) {
		//cout << "split on min gain: " << left_error << " " << right_error << " " << gain << endl;
		class_counts = list_to_discrete(m.column(-1));
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

vector<int> TreeNode::classify(vector<double> & row) {
	if(class_counts.size() > 0) {
		return class_counts;
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
