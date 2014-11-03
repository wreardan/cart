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
	vector<int> class_counts;

public:
	TreeNode();
	~TreeNode();
	void dump(string indent);
	int count();
	void train(Matrix & m, vector<int> columns);
	virtual void train(Matrix & m);
	virtual vector<int> classify(vector<double> & row);
};
