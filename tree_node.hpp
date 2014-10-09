#pragma once

#include "classifier.hpp"
#include "matrix.hpp"

#include <string>
using std::string;

class TreeNode : public Classifier {
private:
	TreeNode * left;
	TreeNode * right;
	int column;
	double value;
	int classification;

public:
	TreeNode();
	~TreeNode();
	void dump(string indent);
	int count();
	void train(Matrix & m, vector<int> columns);
	virtual void train(Matrix & m);
	virtual int classify(vector<double> & row);
};
