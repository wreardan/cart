# Classification and Regression Trees
This software uses regression trees inside a Random Forest to classify
matrices of data.
There are two versions of the software: a Python version in the python folder.  There is a C++ version which is faster and more accurate in the root folder.  Both versions employ parallel programming and run in multiple threads or processes.

# Running the program
To run the program, use the following parameters:
./randomForest -t training_file -c data_to_classify -p 16 -n 1000 -f 30 -m 0.1
## Required Parameters
-t training_file
    input matrix file to train trees
-c data_to_classify
    matrix of data to classify
## Optional Parameters
-p 16
    use 16 threads in a thread pool to process trees
-n 1000
    use 1000 trees in the forest
-f 30
    use a subset of 30 features for each tree
-m 0.1
    minimum gain of sum of squares to continue splitting
    gain = sum_of_squares - (left_squares + right_squares)

# Class Descriptions
- Matrix - 2d double data structure with utility functions
- Classifier - Base class for all classifiers
    - TreeNode - Individual node for a regression tree including the  root node.
    - Forest - Single-threaded Regression Forest Classifier
        - ParallelForest - Multi-threaded training using thread pools

## Other Code Files
- main.cpp - main function and argument processing
- stats.cpp/hpp - statistical functions (mean, mode, variance, regression, etc.)
- util.cpp/hpp - utility functions similar to Python's built-ins
- pthread_pool.c/h - thread pool C code from the Internet

# TODO
    Migrate away from c thread_pool code.  Redo the code in an object-oriented manner.
