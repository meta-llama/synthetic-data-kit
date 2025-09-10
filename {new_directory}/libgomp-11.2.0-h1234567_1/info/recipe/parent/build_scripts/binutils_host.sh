#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

EXTRA_CONFIG="--enable-ld=yes --enable-gold=no"
case "${CFG_ARCH}" in
    arm*)
        EXTRA_CONFIG="--enable-ld=default --enable-gold=yes --enable-threads"
        ;;
esac

rm -rf "${WDIR}/build/binutils-host"
mkdir "${WDIR}/build/binutils-host"
pushd "${WDIR}/build/binutils-host"

    export ac_cv_func_glob=no

    CC_FOR_BUILD="${HOST}-gcc"                                 \
    CFLAGS_FOR_BUILD="${HOST_CFLAG}"                           \
    CXXFLAGS_FOR_BUILD="${HOST_CFLAG}"                         \
    LDFLAGS_FOR_BUILD="${HOST_LDFLAG}"                         \
    CFLAGS="-pipe ${HOST_CFLAG}"                               \
    CXXFLAGS="-pipe ${HOST_CFLAG}"                             \
    LDFLAGS="${HOST_LDFLAG}"                                   \
    bash "${WDIR}/binutils/configure"                          \
        --build=${HOST}                                        \
        --host=${HOST}                                         \
        --target=${CFG_TARGET}                                 \
        --prefix=${WDIR}/gcc_built                             \
        --with-sysroot=${WDIR}/gcc_built/${CFG_TARGET}/sysroot \
        --disable-gdb --disable-multilib                       \
        --disable-nls --disable-sim --disable-werror           \
        --enable-deterministic-archives --enable-plugins       \
        --with-pkgversion="Anaconda binutils"                  \
        ${EXTRA_CONFIG}

    echo "Building binutils ..."
    make

    echo "Installing binutils ..."
    make install

popd

mkdir -p "${WDIR}/gcc_built/lib/bfd-plugins"
mkdir -p "${WDIR}/buildtools/bin"
pushd "${WDIR}/gcc_built/bin"
    for t in "${CFG_TARGET}-"*; do
        if [ "${t}" = "${CFG_TARGET}-*" ]; then
            break
        fi
        _t="${CFG_TARGET}-${t#${CFG_TARGET}-}"
        ln -sfv "${WDIR}/gcc_built/bin/${t}" "${WDIR}/buildtools/bin/${_t}"
    done
popd

mkdir -p "${WDIR}/buildtools/${CFG_TARGET}"
ln -sv "${WDIR}/gcc_built/${CFG_TARGET}/bin" "${WDIR}/buildtools/${CFG_TARGET}/bin"

# clean up ...
rm -rf "${WDIR}/build/binutils-host"

