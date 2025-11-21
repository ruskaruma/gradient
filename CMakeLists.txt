cmake_minimum_required(VERSION 3.20)

project(Gradient VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

find_package(Drogon CONFIG REQUIRED)
find_package(OnnxRuntime CONFIG REQUIRED)
find_package(simdjson CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

file(GLOB_RECURSE SRC_FILES "src/*.cc")

add_executable(gradient ${SRC_FILES})

target_include_directories(gradient PRIVATE src)

target_link_libraries(gradient PRIVATE
    Drogon::Drogon
    OnnxRuntime::OnnxRuntime
    simdjson::simdjson
    fmt::fmt
    gflags::gflags
)

add_custom_command(TARGET gradient POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    $<TARGET_FILE_DIR:gradient>/models
)
