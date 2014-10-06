#include <iostream>
using namespace std;

#include "stats.hpp"
#include "tree_node.hpp"
#include "matrix.hpp"
#include "util.hpp"

int main(int argc, char *argv[]) {
	string filename(argv[1]);
	string output_filename(argv[2]);
	//matrix
	Matrix m;
	m.load(filename);
	//Matrix testing
	Matrix m1, m2;
	m.split(1, 0.001, m1, m2);
	cout << m1.rows() << "\t" << m2.rows() << endl;
	vector<int> r = range(10);
	vector<int> c = range(10);
	Matrix s = m.submatrix(r, c);
	//Regression Tree
	TreeNode root;
	vector<int> columns = range(10);
	root.train(m, columns);
	cout << root.count() << " nodes in tree" << endl;

	//Analyze the results of the tree against training dataset
	int right = 0;
	int wrong = 0;
	for(int i = 0; i < m.rows(); i++) {
		vector<double> & row = m[i];
		int actual_class = root.classify(row);
		int expected_class = row[row.size()-1];
		if(actual_class == expected_class) {
			right++;
		}
		else {
			wrong++;
		}
	}
	double percent = right * 100.0 / m.rows();
	cout << "training set recovered: " << percent << "%" << endl;

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