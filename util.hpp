/*
This file contains utility functions that are similar
to their Python versions, but missing from C++.
*/
#pragma once

#include <vector>
#include <string>
#include <sstream>

//http://stackoverflow.com/questions/10051679/c-tokenize-string
std::vector<std::string> inline split_string(const std::string &source, const char *delimiter = " ", bool keepEmpty = false)
{
    std::vector<std::string> results;

    size_t prev = 0;
    size_t next = 0;

    while ((next = source.find_first_of(delimiter, prev)) != std::string::npos)
    {
        if (keepEmpty || (next - prev != 0))
        {
            results.push_back(source.substr(prev, next - prev));
        }
        prev = next + 1;
    }

    if (prev < source.size())
    {
        results.push_back(source.substr(prev));
    }

    return results;
}

//does somethind similar to Python's ' '.join(list)
template<typename T>
std::string inline join(std::vector<T> & list, const char delimiter) {
    std::stringstream ss;
    for(int i = 0; i < list.size(); i++) {
        ss << list[i];
        if(i < list.size() - 1) {
            ss << delimiter;
        }
    }
    return ss.str();
}

//i.e. double<int> = merge(range(0, 10), range(15, 20));
//immutable
vector<int> inline merge(vector<int> a, vector<int> b) {
    //add second array
    for(int i = 0; i < b.size(); i++) {
        a.push_back(b[i]);
    }
    return a;
}

//Similar to Python's range() function
//Returns a list of integers representing the range
vector<int> inline range(int start, int stop=-1, int step=1) {
    vector<int> result;
    if(stop == -1) {
        stop = start;
        start = 0;
    }
    for(int i = start; i < stop; i += step) {
        result.push_back(i);
    }
    return result;
}

//Convert Python style negative indexes into regular indices
//i.e. (index as int input, list as vector<>)
//index = negative_index_convert(list, index);
template<typename T>
int inline negative_index_convert(std::vector<T> list, int index) {
    //TODO: Bounds checking?
    if(index >= 0) {
        return index;
    }
    else {
        return index + list.size();
    }
}

//Slice an array, immutable, similar to Python's [::] operator
template<typename T>
std::vector<T> inline slice(std::vector<T> list, int start=0, int end=-1, int step=1) {
    vector<T> result;
    start = negative_index_convert(list, start);
    end = negative_index_convert(list, end);
    for(int i = start; i < end; i += step) {
        T element = list[i];
        result.push_back(element);
    }
    return result;
}
