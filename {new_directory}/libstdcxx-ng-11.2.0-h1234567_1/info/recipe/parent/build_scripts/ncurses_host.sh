#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/ncurses-host"
mkdir "${WDIR}/build/ncurses-host"
pushd "${WDIR}/build/ncurses-host"

    CFLAGS="-pipe ${HOST_CFLAG}"           \
    LDFLAGS="${HOST_LDFLAG}"               \
    bash "${WDIR}/ncurses/configure"       \
        --build=${HOST}                    \
        --host=${HOST}                     \
        --prefix="${WDIR}/buildtools"      \
        --with-install-prefix="${destdir}" \
        --without-debug                    \
        --enable-termcap                   \
        --enable-symlinks                  \
        --without-manpages                 \
        --without-tests                    \
        --without-cxx                      \
        --without-cxx-binding              \
        --without-ada                      \
        --disable-database                 \
        --disable-db-install               \
        --with-fallbacks="linux,xterm,xterm-color,xterm-256color,vt100" 

    echo "Building ncurses ..."
    make

    echo "Installing ncurses ..."
    STRIP="${HOST}-strip" make install

popd

# clean up ...
rm -rf "${WDIR}/build/ncurses-host"
