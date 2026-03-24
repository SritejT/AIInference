add_executable(all_tests test_file_parsing.cpp)

target_link_libraries(all_tests GTest::gtest_main)

include(GoogleTest)

gtest_discover_tests(all_tests)
