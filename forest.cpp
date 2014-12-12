#include "forest.hpp"
#include "util.hpp"
#include "stats.hpp"

#include <algorithm>
#include <iostream>
using namespace std;

#include <cstdio>

Forest::Forest() {
	//init(100, 10);  // this is called by child-classes!!
}

Forest::Forest(int n_trees, int n_features) {
	init(n_trees, n_features);
}

void Forest::init(int n_trees, int n_features) {
	//printf("Forest::init(n_trees=%d, n_features=%d)\n", n_trees, n_features);
	this->n_trees = n_trees;
	this->n_features = n_features;
	//Create Trees
	for(int i = 0; i < n_trees; i++) {
		trees.push_back(TreeNode());
	}
}


void Forest::train(Matrix & m) {
	cout << "forest training" << endl;
	vector<int> all_columns = range(m.columns()-1);
	for(int i = 0; i < trees.size(); i++) {
		TreeNode & tree = trees[i];
		random_shuffle(all_columns.begin(), all_columns.end());
		vector<int> subset = slice(all_columns, 0, n_features);
		tree.train(m, subset);
	}
}

int Forest::classify(vector<double> & row) {
	vector<double> summed_distributions(2, 0.0);
	for(int i = 0; i < n_trees; i++) {
		TreeNode & tree = trees[i];
		vector<double> distribution = tree.classify(row);
	}
	return max_index(summed_distributions);
}
