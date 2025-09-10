#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/gcc-core"
mkdir -p "${WDIR}/build/gcc-core"
pushd "${WDIR}/build/gcc-core"

    glibc_version=`echo "${CFG_GLIBC_VER}" | sed 's/\([1-9][0-9]*\.[1-9][0-9]*\).*/\1/'`

    CLANG_CFLAG=
    if ${HOST}-gcc --version 2>&1 | grep clang; then
      CLANG_CFLAG="-fbracket-depth=512"
    fi

    CC_FOR_BUILD="${HOST}-gcc"                                        \
    CFLAGS="${HOST_CFLAG} ${CLANG_CFLAG}"                             \
    CFLAGS_FOR_BUILD="${HOST_CFLAG} ${CLANG_CFLAG}"                   \
    CXXFLAGS="${HOST_CFLAG} ${CLANG_CFLAG}"                           \
    CXXFLAGS_FOR_BUILD="${HOST_CFLAG} ${CLANG_CFLAG}"                 \
    LDFLAGS="${HOST_LDFLAG} -lstdc++ -lm"                             \
    CFLAGS_FOR_TARGET="-g -O2 ${ARCH_CFLAG}"                          \
    CXXFLAGS_FOR_TARGET=""                                            \
    LDFLAGS_FOR_TARGET="${TARGET_LDFLAG}"                             \
    bash "${WDIR}/gcc/configure"                                      \
        --build=${HOST}                                               \
        --host=${HOST}                                                \
        --target=${CFG_TARGET}                                        \
        --prefix="${WDIR}/buildtools"                                 \
        --exec_prefix="${WDIR}/buildtools"                            \
        --with-local-prefix="${WDIR}/gcc_built/${CFG_TARGET}/sysroot" \
        --with-sysroot="${WDIR}/gcc_built/${CFG_TARGET}/sysroot"      \
        --enable-languages="c"                                        \
        --disable-multilib                                            \
        --disable-nls                                                 \
        --with-gmp="${WDIR}/buildtools"                               \
        --with-mpfr="${WDIR}/buildtools"                              \
        --with-mpc="${WDIR}/buildtools"                               \
        --with-isl="${WDIR}/buildtools"                               \
        --without-zstd                                                \
        --with-glibc-version=${glibc_version}                         \
        --with-newlib                                                 \
        --enable-threads=no                                           \
        --disable-shared                                              \
        --enable-__cxa_atexit                                         \
        --disable-libgomp                                             \
        --disable-libmudflap                                          \
        --disable-libmpx                                              \
        --enable-libquadmath                                          \
        --enable-libquadmath-support                                  \
        --disable-libstdcxx-verbose                                   \
        --disable-libstdcxx                                           \
        --with-pkgversion="Anaconda bootstrap gcc"                    \
        --enable-lto

        make configure-gcc configure-libcpp configure-build-libiberty
        make all-libcpp all-build-libiberty
        if [ -d "${WDIR}/gcc/libdecnumber" ]; then
            make configure-libdecnumber
            make -C libdecnumber libdecnumber.a
        fi
        if [ -d "${WDIR}/gcc/libbacktrace" ]; then
            make configure-libbacktrace
            make -C libbacktrace
        fi

    make -C gcc libgcc.mvars
    sed -r -i -e 's@-lc@@g' gcc/libgcc.mvars

    echo "Building gcc core ..."
    CFLAGS="${CFLAGS} -I${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/include"
    make all-gcc all-target-libgcc

    echo "Installing gcc core ..."
    make install-gcc install-target-libgcc

    pushd "${WDIR}/buildtools"
        find . -type f -name "*.la" -exec rm {} \;
    popd

    if [ -f "${WDIR}/buildtools/bin/${CFG_TARGET}-gcc" ]; then
        ln -sfv "${CFG_TARGET}-gcc" "${WDIR}/buildtools/bin/${CFG_TARGET}-cc"
    fi

    multi_root=$( "${WDIR}/buildtools/bin/${CFG_TARGET}-gcc" -print-sysroot )
    if [ ! -e "${multi_root}/lib" ]; then
        ln -sfv lib "${multi_root}/lib"
    fi
    if [ ! -e "${multi_root}/usr/lib" ]; then
        ln -sfv lib "${multi_root}/usr/lib"
    fi

popd

# clean up ...
rm -rf "${WDIR}/build/gcc-core"

