# Inside hnsw_project/benchmarking/Makefile

CC = gcc
CXX = g++
CFLAGS = -O3 -fopenmp
CXXFLAGS = -O3 -std=c++11 -fopenmp -Iwrappers/faiss/faiss

# FAISS source files you want to use
FAISS_DIR = wrappers/faiss/faiss
FAISS_SRC = \
    $(FAISS_DIR)/IndexFlat.cpp \
    $(FAISS_DIR)/IndexIVF.cpp \
    $(FAISS_DIR)/IndexHNSW.cpp \
    $(FAISS_DIR)/IndexIVFPQ.cpp \
    $(FAISS_DIR)/ProductQuantizer.cpp \
    $(FAISS_DIR)/utils.cpp
FAISS_OBJ = indexflat.o utils.o

# Your wrapper
WRAPPER_SRC = wrappers/faiss/faiss_wrapper.cpp
WRAPPER_OBJ = faiss_wrapper.o

# Benchmark code
OBJS = benchmark_main.o benchmark_runner.o benchmark_utils.o $(WRAPPER_OBJ) $(FAISS_OBJ)

# Output binary
TARGET = benchmark

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ -fopenmp -lm -lstdc++

benchmark_main.o: benchmark_main.c
	$(CC) $(CFLAGS) -c $<

benchmark_runner.o: benchmark_runner.c
	$(CC) $(CFLAGS) -c $<

benchmark_utils.o: benchmark_utils.c
	$(CC) $(CFLAGS) -c $<

faiss_wrapper.o: $(WRAPPER_SRC)
	$(CXX) $(CXXFLAGS) -I$(FAISS_DIR) -c $< -o $@

indexflat.o: $(FAISS_DIR)/IndexFlat.cpp
	$(CXX) $(CXXFLAGS) -I$(FAISS_DIR) -c $< -o $@

utils.o: $(FAISS_DIR)/utils.cpp
	$(CXX) $(CXXFLAGS) -I$(FAISS_DIR) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
