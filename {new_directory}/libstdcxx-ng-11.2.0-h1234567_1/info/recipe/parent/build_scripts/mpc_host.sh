#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/mpc-host"
mkdir "${WDIR}/build/mpc-host"
pushd "${WDIR}/build/mpc-host"

    CFLAGS="-pipe ${HOST_CFLAG}"         \
    LDFLAGS="${HOST_LDFLAG}"             \
    bash "${WDIR}/mpc/configure"         \
        --build=${HOST}                  \
        --host=${HOST}                   \
        --prefix="${WDIR}/buildtools"    \
        --with-gmp="${WDIR}/buildtools"  \
        --with-mpfr="${WDIR}/buildtools" \
        --disable-shared                 \
        --enable-static

    echo "Building mpc ..."
    make

    echo "Checking mpc ..."
    make -s check

    echo "Installing mpc ..."
    make install

popd

# clean up ...
rm -rf "${WDIR}/build/mpc-host"

