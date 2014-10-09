#include <iostream>
using namespace std;
#include <cstdlib>
#include <cassert>
#include <unistd.h>


#include "stats.hpp"
#include "tree_node.hpp"
#include "matrix.hpp"
#include "util.hpp"
#include "forest.hpp"
#include "parallel_forest.hpp"

int threads = 16;
int n_trees = 1000;
int n_features = -1;

double test(Classifier * c, Matrix & m, vector<int> & classes) {
	classes.empty();
	//Analyze the results of the tree against training dataset
	int right = 0;
	int wrong = 0;
	for(int i = 0; i < m.rows(); i++) {
		vector<double> & row = m[i];
		int actual_class = c->classify(row);
		classes.push_back(actual_class);
		int expected_class = row[row.size()-1];
		if(actual_class == expected_class) {
			right++;
		}
		else {
			wrong++;
		}
	}
	double percent = right * 100.0 / m.rows();
	return percent;
}

void train_and_test(Matrix & matrix) {
	vector<Classifier*> classifiers;
	vector<int> classes;
	//Other classifier commented out:
	//classifiers.push_back(new TreeNode());
	//classifiers.push_back(new Forest(1000, matrix.columns()-1));

	//use all features if negative
	if(n_features <= 0) {
		n_features = matrix.columns()-1;
	}
	classifiers.push_back(new ParallelForest(n_trees, n_features, threads));

	for(int i = 0; i < classifiers.size(); i++) {
		Classifier * classifier = classifiers[i];
		//cout << "training classifier #" << i << endl;
		classifier->train(matrix);
		double percent = test(classifier, matrix, classes);
		cout << "training set recovered: " << percent << "%" << endl;
	}
}

void test_matrix(Matrix & m) {
	//Matrix testing
	Matrix m1, m2;
	m.split(1, 0.001, m1, m2);
	cout << m1.rows() << "\t" << m2.rows() << endl;
	vector<int> r = range(10);
	vector<int> c = range(10);
	Matrix s = m.submatrix(r, c);
	//Regression Tree
	TreeNode root;
	vector<int> columns = range(m.columns()-1);
	root.train(m, columns);
	cout << root.count() << " nodes in tree" << endl;
}

void test_stats() {
	//stats
	test_regression();
	vector<double> test_mode;
	test_mode.push_back(1.0);
	test_mode.push_back(2.0);
	test_mode.push_back(5.0);
	test_mode.push_back(2.0);
	test_mode.push_back(7.0);
	cout << mode(test_mode) << endl;
}

int main(int argc, char *argv[]) {
	//Parse command line arguments
	char * cvalue = NULL;
	int c;
	string filename;
	while((c = getopt(argc, argv, "t:c:p:n:f:m:")) != -1) {
		switch(c) {
		case 't': //training file
			filename = optarg;
			break;
		case 'c': //data to classify
			break;
		case 'p': //number of threads
			threads = atoi(optarg);
			break;
		case 'n': //number of trees
			n_trees = atoi(optarg);
			assert(n_trees > 0);
			break;
		case 'f': //number of features
			n_features = atoi(optarg);
			assert(n_features > 0);
			break;
		case 'm': //minimum gain
			break;
		default:
			exit(1);
		}
	}
	if(threads <= 0) threads = 16;
	cout << "using " << threads << " threads in pool" << endl;
	//matrix
	Matrix m;
	m.load(filename);
	cout << m.rows() << " rows and " << m.columns() << " columns in matrix" << endl;

	//Run classifiers
	train_and_test(m);

	return 0;
}
