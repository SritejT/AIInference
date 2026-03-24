add_executable(all_tests test_file_parsing.cpp)

target_link_libraries(all_tests GTest::gtest_main AIInference_lib)

target_compile_definitions(all_tests PRIVATE
  SRC_DIR="${CMAKE_SOURCE_DIR}"
)

include(GoogleTest)

gtest_discover_tests(all_tests)
