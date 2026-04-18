cmake_minimum_required(VERSION 3.20)
project(AIInference LANGUAGES CXX)

option(ENABLE_BENCHMARKS "Build benchmarks" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
add_compile_options(-fno-omit-frame-pointer)
add_compile_options(-march=armv8.4-a+sve)

file(GLOB_RECURSE lib_sources "src/*.cpp")

add_library(AIInference_lib STATIC ${lib_sources})

target_include_directories(AIInference_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE AIInference_lib)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -Wpedantic
)

Include(FetchContent)

add_subdirectory(tests)

if (ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()


