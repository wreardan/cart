#pragma once

#include "classifier.hpp"
#include "matrix.hpp"
#include "tree_node.hpp"

#include <vector>
using std::vector;

class Forest : public Classifier {
protected:
	int n_trees;
	int n_features;
	vector<TreeNode> trees;
public:
	Forest();
	Forest(int n_trees, int n_features);
	void init(int n_trees, int n_features);
	virtual void train(Matrix & m);
	virtual int classify(vector<double> & row);
	virtual double soft_classify(vector<double> & row);
};
