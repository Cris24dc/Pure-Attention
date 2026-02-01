#!/bin/bash

nvcc -O3 -arch=sm_61 main.cu -o matmul_bench.o
./matmul_bench.o > results.csv
python3 matmul_test.py