#include <iostream>
#include <cstdlib>
using namespace std;

#include "stats.hpp"
#include "tree_node.hpp"
#include "matrix.hpp"
#include "util.hpp"
#include "forest.hpp"
#include "parallel_forest.hpp"

int threads;

double test(Classifier * c, Matrix & m) {
	//Analyze the results of the tree against training dataset
	int right = 0;
	int wrong = 0;
	for(int i = 0; i < m.rows(); i++) {
		vector<double> & row = m[i];
		int actual_class = c->classify(row);
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
	//classifiers.push_back(new TreeNode());
	//classifiers.push_back(new Forest(1000, matrix.columns()-1));
	classifiers.push_back(new ParallelForest(10, matrix.columns()-1, threads));

	for(int i = 0; i < classifiers.size(); i++) {
		Classifier * classifier = classifiers[i];
		cout << "training classifier #" << i << endl;
		classifier->train(matrix);
		double percent = test(classifier, matrix);
		cout << "training set recovered: " << percent << "%" << endl;
	}
}

int main(int argc, char *argv[]) {
	string filename(argv[1]);
	threads = atoi(argv[2]);
	if(threads <= 0) threads = 16;
	cout << " threads " << threads << endl;
	//matrix
	Matrix m;
	m.load(filename);
	cout << m.rows() << " rows and " << m.columns() << " columns in matrix" << endl;
	//Matrix testing
	/*
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


	//Random Forest
	Forest f(100, 20);
	f.train(m);
*/
	//Run classifiers
	train_and_test(m);

	//stats
	test_regression();
	vector<double> test_mode;
	test_mode.push_back(1.0);
	test_mode.push_back(2.0);
	test_mode.push_back(5.0);
	test_mode.push_back(2.0);
	test_mode.push_back(7.0);
	cout << mode(test_mode) << endl;
	return 0;
}