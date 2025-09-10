#!/bin/bash

# Create and enter the build directory
mkdir -p build && cd build

# Build shared library
cmake -GNinja \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      .. || exit 1

ninja && ninja install