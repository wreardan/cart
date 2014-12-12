#pragma once

#include "classifier.hpp"
#include "matrix.hpp"

#include <string>
using std::string;

extern double MINIMUM_GAIN;

class TreeNode {
private:
	TreeNode * left;
	TreeNode * right;
	int column;
	double value;
	double gain;
	vector<double> distribution;

//Helper methods:
	void split_xy(const vector<double> & X, const vector<double> & Y, double value, vector<double> & lesser, vector<double> & greater);
	void train_gini(Matrix & m, vector<int> columns, int n_columns=7, int n_classes=2);

public:
	TreeNode();
	~TreeNode();
	void dump(string indent);
	int count();
	void train(Matrix & m, vector<int> columns);
	virtual void train(Matrix & m);
	virtual vector<double> classify(vector<double> & row);
};
