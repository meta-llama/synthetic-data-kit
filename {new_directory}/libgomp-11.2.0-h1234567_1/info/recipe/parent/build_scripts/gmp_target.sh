#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

GMP_HOST=$CFG_TARGET
OSX_CONFIG=
case "${CFG_TARGET}" in
    *darwin*)
        OSX_CONFIG="--with-pic"
        ;;
    *power*)
        GMP_HOST="power8-pc-linux-gnu"
        ;;
esac

rm -rf "${WDIR}/build/gmp-target"
mkdir "${WDIR}/build/gmp-target"
pushd "${WDIR}/build/gmp-target"

    CC="${CFG_TARGET}-gcc"                    \
    CXX="${CFG_TARGET}-g++"                   \
    CFLAGS="${ARCH_CFLAG} -fexceptions"       \
    CXXFLAGS="${ARCH_CFLAG}"                  \
    LDFLAGS="${TARGET_LDFLAG} ${ARCH_LDFLAG}" \
    bash "${WDIR}/gmp/configure"              \
        --build=${HOST}                       \
        --host=${GMP_HOST}                    \
        --prefix=/usr                         \
        --enable-fft                          \
        --enable-cxx                          \
        --enable-shared                       \
        --enable-static ${OSX_CONFIG}

    echo "Building gmp ..."
    make

    # don't test here as C++ runtime is not working
    # for built toolchain at this point

    echo "Installing gmp ..."
    make install DESTDIR="${WDIR}/gcc_built/${CFG_TARGET}/sysroot"

popd

# clean up ...
rm -rf "${WDIR}/build/gmp-target"

