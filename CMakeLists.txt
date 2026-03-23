cmake_minimum_required(VERSION 3.20)
project(AIInference LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/tensors.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -Wpedantic
)
