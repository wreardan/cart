#pragma once

#include <vector>
using std::vector;
#include <cassert>
#include <cstdlib>

double sum(const vector<double> & list);
double sum_squared(const vector<double> & list);
double mean(const vector<double> & list);
double mode(const vector<double> & list);
double covariance(const vector<double> & dist1, const vector<double> & dist2);
void basic_linear_regression(const vector<double> & x, const vector<double> & y, double & m, double & b);
double sum_of_squares(const vector<double> & x, const vector<double> & y, double m, double b);
void test_regression();

vector<int> list_to_discrete(const vector<double> & list, int num_classes=2);
void add_counts(vector<int> & list1, vector<int> & list2);

//Template Functions (must be in header)

/*
vec = [0,1,2,3]
max_index(vec) = 3
*/
template<typename T>
int max_index(vector<T> list) {
    assert(list.size() > 0);
    int index = 0;
    T biggest = list[index];
    for(int i = 1; i < list.size(); i++) {
        T value = list[i];
        if(value > biggest) {
            index = i;
            biggest = value;
        }
    }
}

/*
Sample from a list without replacement
puts the result in results_list
*/
template<typename T>
void sample(vector<T> sampling_list, int n, vector<T> & result_list) {
    random_shuffle(sampling_list.begin(), sampling_list.end());
    assert(n < sampling_list.size());
    result_list.empty();
    for(int i = 0; i < n; i++) {
        result_list.push_back(sampling_list[i]);
    }
}


/*
Sample from a list with replacement
puts the result in results_list
*/
template<typename T>
void sample_with_replacement(vector<T> sampling_list, int n, vector<T> & result_list) {
    result_list.empty();
    int size = sampling_list.size();
    for(int i = 0; i < n; i++) {
        int index = rand() % size;
        result_list.push_back(sampling_list[index]);
    }
}
