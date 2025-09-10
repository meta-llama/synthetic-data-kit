

set -ex



xz --help
unxz --help
lzma --help
test -f ${PREFIX}/include/lzma.h
test -f ${PREFIX}/lib/cmake/liblzma/liblzma-config.cmake
test -f ${PREFIX}/lib/pkgconfig/liblzma.pc
test -f `pkg-config --variable=libdir --dont-define-prefix liblzma`/liblzma${SHLIB_EXT}
test ! -f ${PREFIX}/lib/liblzma.a
test -f ${PREFIX}/lib/liblzma${SHLIB_EXT}
test -f ${PREFIX}/lib/liblzma${SHLIB_EXT}.5
test -f ${PREFIX}/lib/liblzma${SHLIB_EXT}.5.6.4
conda inspect linkages -p $PREFIX $PKG_NAME
exit 0
