
# Setup googletest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)


# Create test executable
add_executable(all_tests test_file_parsing.cpp test_matrix_mult.cpp test_matrix_add.cpp test_threadpool.cpp
    test_matrix_transpose.cpp)

target_link_libraries(all_tests GTest::gtest_main AIInference_lib)

target_compile_definitions(all_tests PRIVATE
  SRC_DIR="${CMAKE_SOURCE_DIR}"
)


gtest_discover_tests(all_tests)
