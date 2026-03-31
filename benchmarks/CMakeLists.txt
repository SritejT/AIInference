
# Setup google benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)


# Create benchmark executable
add_executable(all_benchmarks benchmark_matrix_mult.cpp)
target_link_libraries(all_benchmarks PRIVATE AIInference_lib benchmark::benchmark)

target_compile_definitions(all_benchmarks PRIVATE
  SRC_DIR="${CMAKE_SOURCE_DIR}"
)

