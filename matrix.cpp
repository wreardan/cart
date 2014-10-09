#include "matrix.hpp"
#include "util.hpp"

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <string>
#include <iostream>
using namespace std;

Matrix::Matrix() {
	
}

void Matrix::load(string filename, bool use_column_labels, bool use_row_labels) {
	ifstream file(filename.c_str());
	string line;
	int line_number = 0;
	while(getline(file, line)) {
		vector<string> tokens = split_string(line, "\t");
		if(tokens.size() == 0) {
			cout << "Matrix.load(): skipping blank line on line #" << line_number << endl;
			line_number++;
			continue;
		}
		if(use_column_labels && line_number == 0) {
			column_labels = tokens;
			line_number++;
			continue;
		}
		else {
			if(use_row_labels) {
				row_labels.push_back(tokens[0]);
				tokens.erase(tokens.begin()+0);
			}
			vector<double> row;
			for(int i = 0; i < tokens.size(); i++) {
				double element = atof(tokens[i].c_str());
				row.push_back(element);
			}
			elements.push_back(row);
		}
		line_number++;
	}
	file.close();
}

void Matrix::save(string filename) {
	ofstream file(filename.c_str());
	//Write column header
	if(column_labels.size() > 0) {
		file << join(column_labels, '\t');
	}
	file << endl;
	//Write elements
	for(int i = 0; i < elements.size(); i++) {
		vector<double> & row = elements[i];
		if(row_labels.size() > 0) {
			file << row_labels[i] << '\t';
		}
		file << join(row, '\t');
		file << endl;
	}
	file.close();
}

int Matrix::columns() {
	if(elements.size() == 0) {
		return 0;
	}
	else {
		return elements[0].size();
	}
}

int Matrix::rows() {
	return elements.size();
}

vector<double> & Matrix::operator[](int i) {
	assert(i < elements.size());
	return elements[i];
}

vector<double> Matrix::column(int index) {
	vector<double> result;
	if(index < 0) {
		index += columns();
	}
	for(int i = 0; i < rows(); i++) {
		double element = elements[i][index];
		result.push_back(element);
	}
	return result;
}

/*
Return a subset of the matrix's elements.
rows is a list of row indices
columns is a list of column indices
i.e.
m = [[0,1],[2,3],[4,5]]
m.submatrix([0,1],[1]) = [[1],[3]]
*/
Matrix Matrix::submatrix(vector<int> rows, vector<int> columns) {
	Matrix m;
	for(int j = 0; j < columns.size(); j++) {
		m.column_labels.push_back(column_labels[j]);
	}
	for(int i = 0; i < rows.size(); i++) {
		int y = rows[i];
		vector<double> row;
		m.row_labels.push_back(row_labels[y]);
		for(int j = 0; j < columns.size(); j++) {
			int x = columns[j];
			row.push_back(elements[y][x]);
		}
		m.elements.push_back(row);
	}
	return m;
}

/*
Split the matrix into two based on a column's value
i.e.
m = [[0,1],[0,2],[0,3]]
Matrix m1, m2;
m.split(1, 2, m1, m2);
m1 = [[0,1]]
m2 = [[0,2],[0,3]]
*/
void Matrix::split(int column_index, double value, Matrix & m1, Matrix & m2) {
	vector<int> m1_rows;
	vector<int> m2_rows;
	for(int i = 0; i < elements.size(); i++) {
		double element = elements[i][column_index];
		if(element < value) {
			m1_rows.push_back(i);
		}
		else {
			m2_rows.push_back(i);
		}
	}
	vector<int> all_cols = range(columns());
	m1 = submatrix(m1_rows, all_cols);
	m2 = submatrix(m2_rows, all_cols);
}

