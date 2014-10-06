build:
	g++ -pthread -std=c++0x main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest
	time ./randomForest ../data/H1hesc_allforward_features.txt ../data/test.txt

debug:
	g++ -g -pthread -std=c++0x main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest
	gdb ./randomForest