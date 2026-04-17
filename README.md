## Summary

AIInference aims to be a high-performance, minimal dependency machine learning library written in C++. The long-term goal
is to provide efficient implementations of core ML algorithms and primitives from scratch.

Currently, the library has parallelised implementations of matrix addition, multiplication, inversion, transposition, and
application of unary operations (e.g. multiplication / division by scalars).

## Get Started

**Prerequisites:** CMake 3.20+, a C++20-compatible compiler (e.g. Clang or GCC), and an internet connection for the first build (Google Test and Google Benchmark are fetched automatically).

**Build and run:**

```bash
git clone https://github.com/SritejT/AIInference.git
cd AIInference
cmake -S . -B build
cmake --build build
./build/AIInference
```

**Run unit tests (GTest):**

```bash
ctest --test-dir build/tests --output-on-failure
```

For convenience, use the provided script to build and test the project in one go:

```bash
bash run.sh
```

**Run benchmarks (Google Benchmark):**

```bash
./build/mult_benchmarks
./build/add_benchmarks
./build/transpose_benchmarks
./build/inverse_benchmarks
```

## Performance

Uses ARM NEON/SVE SIMD intrinsics for data level parallelism, and a custom threadpool implementation for task level 
parallelism. Also uses simple blocked loops for better cache locality.

It achieves up to 380x speedup over a naive implementation for multiplying 2048x2048 float32 matrices (around 170 GFLOPS
on an AWS EC2 c7g.4xlarge instance). It's around 5x slower than common BLAS libraries such as OpenBLAS or MKL, but I 
plan to close this gap in the coming months. 

Feel free to run the google benchmarks to see for yourself! I've put screenshots of some of the results below:

### Matrix Addition

**Naive implementation:**

![Basic tensor addition benchmarks](images/BasicTensorAdd.png)

**Optimised implementation:**

![Optimised tensor addition benchmarks](images/OptimisedTensorAdd.png)

### Matrix Multiplication

**Naive implementation:**

![Basic tensor multiplication benchmarks](images/BasicTensorMult.png)

**Optimised implementation:**

![Optimised tensor multiplication benchmarks](images/OptimisedTensorMult.png)

## Coming Soon

- **Further optimisations** - I'm planning to experiment with different loop blocking techniques, threadpool implementations,
and SIMD vectorisation.
- **Linear models** — Once I have a sufficiently fast matrix implementation, this should be pretty straightforward :) 
- **Neural nets** - I have written a basic safetensors parser, but it currently relies on rapidjson to parse the header.
I plan to replace this with a custom parser that can parse safetensors directly, and then make a wrapper for the neural net API.
Again, should be straightforward with a good matrix implementation.

