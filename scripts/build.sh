#!/bin/bash
set -e

if [ ! -d "vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git
    ./vcpkg/bootstrap-vcpkg.sh
fi

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build build --config Release -j$(nproc)
