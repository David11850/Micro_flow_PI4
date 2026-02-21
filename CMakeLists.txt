cmake_minimum_required(VERSION 3.16)
project(MicroFlow
    VERSION 2.0
    DESCRIPTION "Lightweight Inference Engine for Raspberry Pi 4"
    LANGUAGES CXX
)

# 1. 基础环境设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 2. 定义微流核心库 (必须放在 target_... 指令之前)
# ============================================================================
add_library(microflow_core STATIC
    src/memory/allocator.cpp
    src/memory/tensor.cpp
    src/memory/image.cpp
    src/gemm/gemm.cpp
    src/conv/conv.cpp
    src/layers/layers.cpp
    src/runtime/runtime.cpp
)

# 3. 硬件架构专项优化 (针对树莓派4 Cortex-A72)
# ============================================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    message(STATUS "Configuring for Raspberry Pi 4 (ARM64)")

    target_compile_options(microflow_core PRIVATE
        -march=armv8-a
        -mtune=cortex-a72
        -mcpu=cortex-a72
        -O3
        -ffast-math
        -funsafe-math-optimizations
        -ffp-contract=fast
        -funroll-loops
        -ftree-vectorize
        -flto
    )
    target_link_options(microflow_core PRIVATE -flto)
    set(DEFAULT_NUM_THREADS 4)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Configuring for x86_64 (development)")
    target_compile_options(microflow_core PRIVATE -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize)
    set(DEFAULT_NUM_THREADS 4)
endif()

# 4. 依赖项与编译器配置
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(microflow_core PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# 查找并配置 OpenMP (多线程支持)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    target_link_libraries(microflow_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# 包含目录
target_include_directories(microflow_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/microflow>
    $<INSTALL_INTERFACE:include>
)

# 5. 可执行测试与示例程序
# ============================================================================
# 性能基准测试
add_executable(benchmark tests/benchmark.cpp)
target_link_libraries(benchmark microflow_core)

# MNIST推理示例
add_executable(mnist_demo examples/mnist_demo.cpp)
target_link_libraries(mnist_demo microflow_core)

# 图像推理示例
add_executable(image_inference examples/image_inference.cpp)
target_link_libraries(image_inference microflow_core)

# 图像加载演示 (基于mnist_demo)
add_executable(image_demo examples/image_demo.cpp)
target_link_libraries(image_demo microflow_core)

# 图像预处理工具
add_executable(image_preprocess examples/image_preprocess.cpp)
target_link_libraries(image_preprocess microflow_core)

# 图像加载测试
add_executable(test_image tests/test_image.cpp)
target_link_libraries(test_image microflow_core)

# 基础测试组件
#add_executable(test_allocator tests/test_allocator.cpp)
#target_link_libraries(test_allocator microflow_core)

#add_executable(test_gemm tests/test_gemm.cpp)
#target_link_libraries(test_gemm microflow_core)

# 6. 安装配置
# ============================================================================
install(TARGETS microflow_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/microflow
)
