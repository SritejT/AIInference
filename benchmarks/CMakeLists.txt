
# Setup google benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)


# Create benchmark executables
add_executable(mult_benchmarks benchmark_matrix_mult.cpp)
target_link_libraries(mult_benchmarks PRIVATE AIInference_lib benchmark::benchmark)

add_executable(add_benchmarks benchmark_matrix_add.cpp)
target_link_libraries(add_benchmarks PRIVATE AIInference_lib benchmark::benchmark)

add_executable(transpose_benchmarks benchmark_matrix_transpose.cpp)
target_link_libraries(transpose_benchmarks PRIVATE AIInference_lib benchmark::benchmark)

add_executable(inverse_benchmarks benchmark_matrix_inverse.cpp)
target_link_libraries(inverse_benchmarks PRIVATE AIInference_lib benchmark::benchmark)
