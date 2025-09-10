#! /bin/bash

set -e
set -x

# Adopt a Unix-friendly path if we're on Windows (see bld.bat).
[ -n "$PATH_OVERRIDE" ] && export PATH="$PATH_OVERRIDE"

# On Windows we want $LIBRARY_PREFIX in both "mixed" (C:/Conda/...) and Unix
# (/c/Conda) forms, but Unix form is often "/" which can cause problems.
if [ -n "$LIBRARY_PREFIX_M" ] ; then
    mprefix="$LIBRARY_PREFIX_M"
    if [ "$LIBRARY_PREFIX_U" = / ] ; then
        uprefix=""
    else
        uprefix="$LIBRARY_PREFIX_U"
    fi
else
    mprefix="$PREFIX"
    uprefix="$PREFIX"
fi

# On Windows we need to regenerate the configure scripts.
if [ -n "$CYGWIN_PREFIX" ] ; then
    export ACLOCAL=aclocal-$am_version
    export AUTOMAKE=automake-$am_version
    autoreconf_args=(
        --force
        --install
        -I "$mprefix/share/aclocal"
        -I "$BUILD_PREFIX_M/Library/usr/share/aclocal"
    )
    autoreconf "${autoreconf_args[@]}"

    # ./configure needs this
    export CPP=x86_64-w64-mingw32-cpp.exe

    # Asking for this leads to compile errors:
    export CONFIG_FLAGS="--disable-ipv6"

    # And we need to add the search path that lets libtool find the
    # msys2 stub libraries for ws2_32.
    # Look in standard mingw-w64 library locations
    for lib_path in "$BUILD_PREFIX_M/Library/mingw-w64/bin" "$BUILD_PREFIX_M/Library/bin" "$mprefix/Library/mingw-w64/lib" "$mprefix/Library/lib"; do
        if [ -d "$lib_path" ]; then
            # Convert to Windows path
            win_path=$(cygpath -w "$lib_path")
            echo "Adding path to library search: $win_path"
            export PATH="$PATH:$win_path"
            export LIBRARY_PATH="$LIBRARY_PATH:$win_path"
            # For ld
            export LDFLAGS="$LDFLAGS -L$win_path"
        fi
    done

    # Explicitly set preprocessor to use gcc in preprocessing mode
    export CPP="gcc -E"
    
    # Ensure compiler path is correct
    echo "Using compiler: $CC"
    echo "Path: $PATH"
    
    # Add configure flags to help with Windows builds
    configure_args+=(
        --build=x86_64-w64-mingw32
        --host=x86_64-w64-mingw32
    )

    # Needed to get X11/X.h
    export CFLAGS="$CFLAGS -I$LIBRARY_PREFIX_U/include"
else
    # for other platforms we just need to reconf to get the correct achitecture
    libtoolize --force
    aclocal -I $PREFIX/share/aclocal -I $BUILD_PREFIX/share/aclocal
    autoheader
    autoconf
    automake --force-missing --add-missing --include-deps

    export CONFIG_FLAGS="--build=${BUILD}"
fi

if [[ "$(uname)" == "Darwin" ]]; then
    export CPP=clang-cpp
    ln -s $BUILD_PREFIX/bin/clang-cpp $BUILD_PREFIX/bin/cpp
fi

export PKG_CONFIG_LIBDIR=$uprefix/lib/pkgconfig:$uprefix/share/pkgconfig
configure_args=(
    $CONFIG_FLAGS
    --prefix=$mprefix
    --disable-static
    --disable-dependency-tracking
    --disable-selective-werror
    --disable-silent-rules
)

# Unix domain sockets aren't gonna work on Windows
if [ -n "$CYGWIN_PREFIX" ] ; then
    configure_args+=(--disable-unix-transport)
fi

if [[ "${CONDA_BUILD_CROSS_COMPILATION}" == "1" ]]; then
    export xorg_cv_malloc0_returns_null=yes
fi

./configure "${configure_args[@]}"
make -j$CPU_COUNT
make install
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" != "1" || "${CROSSCOMPILING_EMULATOR}" != "" ]]; then
make check
fi

rm -rf $uprefix/share/doc/libX11 $uprefix/share/man

# Remove any new Libtool files we may have installed. It is intended that
# conda-build will eventually do this automatically.
find $uprefix/. -name '*.la' -delete
