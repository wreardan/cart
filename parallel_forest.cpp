#include "parallel_forest.hpp"
#include "util.hpp"

#include "pthread_pool.h"

#include <algorithm>
#include <iostream>
using namespace std;

ParallelForest::ParallelForest() {
	init(100, 10);
	n_threads = 4;
}

ParallelForest::ParallelForest(int n_trees, int n_features, int n_threads) {
	this->n_threads = n_threads;
	init(n_trees, n_features);
}

struct Work {
	Matrix * matrix;
	TreeNode * tree;
	vector<int> * subset;
};

void * training_thread(void * void_ptr) {
	Work * work = (Work*)void_ptr;
	work->tree->train(*work->matrix, *work->subset);
	return NULL;
}

void ParallelForest::train(Matrix & m) {
	cout << "parallel forest training" << endl;
	//Create thread pool
	void * pool = pool_start(&training_thread, n_threads);
	//Run through threads
	vector<vector<int> > all_subsets(trees.size());
	vector<int> all_columns = range(m.columns()-1);
	for(int i = 0; i < trees.size(); i++) {
		TreeNode & tree = trees[i];
		random_shuffle(all_columns.begin(), all_columns.end());
		all_subsets[i] = slice(all_columns, 0, n_features);

		//create work
		struct Work * work = new struct Work;
		work->matrix = &m;
		work->tree = &tree;
		work->subset = &all_subsets[i];

		pool_enqueue(pool, work, true);
	}
	//JOIN on all
	pool_wait(pool);
}

