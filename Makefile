test: build
	time ./randomForest -t ./data/H1hesc_allforward_features.txt -p 4 -n 100

build:
	g++ -pthread main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest


optimized:
	g++ -pthread -O3 -D NDEBUG main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest

debug:
	g++ -g -pthread main.cpp stats.cpp tree_node.cpp matrix.cpp forest.cpp parallel_forest.cpp pthread_pool.c -o randomForest

debugger: debug
	gdb --args ./randomForest -t ./data/H1hesc_allforward_features.txt -p 4 -n 100

valgrind: debug
	valgrind --leak-check=full ./randomForest -t ./data/H1hesc_allforward_features.txt -p 4 -n 100
