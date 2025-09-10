#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

case "${HOST}" in
    *linux*)
        exit 0
        ;;
esac

rm -rf "${WDIR}/build/libiconv-host"
mkdir "${WDIR}/build/libiconv-host"
pushd "${WDIR}/build/libiconv-host"

    CFLAGS="-pipe ${HOST_CFLAG}"      \
    LDFLAGS="${HOST_LDFLAG}"          \
    bash "${WDIR}/libiconv/configure" \
        --build=${HOST}               \
        --host="${HOST}"              \
        --prefix="${WDIR}/buildtools" \
        --enable-static               \
        --disable-shared              \
        --disable-nls

    echo "Building libiconv ..."
    make CC="${HOST}-gcc -pipe ${HOST_CFLAG}"

    echo "Installing libiconv ..."
    make CC="${HOST}-gcc -pipe ${HOST_CFLAG}" install

popd

# clean up ...
rm -rf "${WDIR}/build/libiconv-host"

