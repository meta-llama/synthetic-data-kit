#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/libelf-target"
mkdir -p "${WDIR}/build/libelf-target"
pushd "${WDIR}/build/libelf-target"

    CC="${CFG_TARGET}-gcc"          \
    RANLIB="${CFG_TARGET}-ranlib"   \
    CFLAGS="${ARCH_CFLAG} -fPIC"    \
    LDFLAGS="${TARGET_LDFLAG}"      \
    bash "${WDIR}/libelf/configure" \
        --build=${HOST}             \
        --host=${CFG_TARGET}        \
        --target=${CFG_TARGET}      \
        --prefix=/usr               \
        --enable-compat             \
        --enable-elf64              \
        --enable-extended-format    \
        --enable-static             \
        --enable-shared

    echo "Building libelf ..."
    make

    echo "Installing libelf ..."
    make instroot="${WDIR}/gcc_built/${CFG_TARGET}/sysroot" install
popd

# clean up ...
rm -rf "${WDIR}/build/libelf-target"

