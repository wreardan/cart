#include "forest.hpp"
#include "util.hpp"
#include "stats.hpp"

#include <algorithm>
#include <iostream>
using namespace std;

Forest::Forest() {
	init(100, 10);
}

Forest::Forest(int n_trees, int n_features) {
	init(n_trees, n_features);
}

void Forest::init(int n_trees, int n_features) {
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
	vector<double> votes;
	for(int i = 0; i < n_trees; i++) {
		TreeNode & tree = trees[i];
		double vote = tree.classify(row);
		votes.push_back(vote);
	}
	return (int)mode(votes);
}

