build:
	g++ -pthread main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest
	time ./randomForest ../data/H1hesc_allforward_features.txt 16

debug:
	g++ -g -pthread -std=c++0x main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest

debugger: debug
	gdb ./randomForest

valgrind: debug
	valgrind ./randomForest ../data/H1hesc_allforward_features.txt 2
