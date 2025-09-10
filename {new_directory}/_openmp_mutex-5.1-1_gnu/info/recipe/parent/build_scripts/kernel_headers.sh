#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/kernel-headers"
mkdir -p "${WDIR}/build/kernel-headers"
pushd "${WDIR}/build/kernel-headers"

    case "${CFG_ARCH}:64" in
        arm:64) kernel_arch="arm64";;
        *) kernel_arch="${CFG_ARCH}";;
    esac

    echo "Installing kernel headers"
    make -C "${WDIR}/linux"                                            \
        BASH="$(which bash)"                                           \
        HOSTCC="${HOST}-gcc"                                           \
        CROSS_COMPILE="${CFG_TARGET}-"                                 \
        O="${WDIR}/build/kernel-headers"                               \
        ARCH=${kernel_arch}                                            \
        INSTALL_HDR_PATH="${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr" \
        V=0                                                            \
        headers_install

    echo "Checking installed headers"
    make -C "${WDIR}/linux"                                            \
        BASH="$(which bash)"                                           \
        HOSTCC="${HOST}-gcc"                                           \
        CROSS_COMPILE="${CFG_TARGET}-"                                 \
        O="${WDIR}/build/kernel-headers"                               \
        ARCH=${kernel_arch}                                            \
        INSTALL_HDR_PATH="${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr" \
        V=0                                                            \
        headers_check

    find "${WDIR}/gcc_built/${CFG_TARGET}/sysroot" -type f \
        \( -name '.install' -o -name '..install.cmd' -o -name '.check' -o -name '..check.cmd' \) -exec rm {} \;

popd

# clean up ...
rm -rf "${WDIR}/build/kernel-headers"
