/*
An Abstract Base Class (cannot instantiate directly)
for Machine Learning algorithms that classify data
*/
#pragma once

#include "matrix.hpp"

#include <iostream>

class Classifier {
public:
	virtual void train(Matrix & m) {
		std::cout << "classifier no training";
	};
	virtual int classify(vector<double> & row) {return 0;};
};
