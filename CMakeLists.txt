cmake_minimum_required(VERSION 3.20)
project(AIInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
add_compile_options(-fno-omit-frame-pointer)
add_compile_options(-march=armv8.4-a+simd)

add_library(AIInference_lib STATIC
    src/tensors/tensor.cpp
    src/tensors/basic_tensor.cpp
    src/tensors/fast_tensor.cpp
    src/tensors/strategies/basic_tensor_strategy.cpp
    src/parse_safetensors.cpp
)

target_include_directories(AIInference_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}
    src/main.cpp
)

add_executable(base
    src/run_basic_tensors.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE AIInference_lib)
target_link_libraries(base PRIVATE AIInference_lib)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -Wpedantic
)

target_compile_options(base PRIVATE
    -Wall -Wextra -Wpedantic
)


Include(FetchContent)

add_subdirectory(tests)
add_subdirectory(benchmarks)


