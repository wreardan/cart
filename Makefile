build:
	g++ -pthread main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest
	time ./randomForest -t ./data/H1hesc_allforward_features.txt -p 4 -n 100

optimized:
	g++ -pthread -O3 -D NDEBUG main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest

debug:
	g++ -g -pthread main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest

debugger: debug
	gdb ./randomForest

valgrind: debug
	valgrind --leak-check=full ./randomForest ./data/H1hesc_allforward_features.txt 2
