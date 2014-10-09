#pragma once

#include "forest.hpp"

class ParallelForest : public Forest {
protected:
	int n_threads;
public:
	ParallelForest();
	ParallelForest(int n_trees, int n_features, int n_threads);
	virtual void train(Matrix & m);
};