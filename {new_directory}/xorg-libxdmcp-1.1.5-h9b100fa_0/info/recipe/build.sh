#! /bin/bash

set -xe

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

    export CC="gcc"

    # Look in standard mingw-w64 library locations
    platlibs=""
    for potential_path in \
        "$(dirname $($CC --print-prog-name=ld))/../sysroot/usr/lib" \
        "$(dirname $($CC --print-prog-name=ld))/../x86_64-w64-mingw32/lib" \
        "$BUILD_PREFIX_M/Library/mingw-w64/lib" \
        "$BUILD_PREFIX_M/Library/usr/lib" \
        "$BUILD_PREFIX_M/Library/x86_64-w64-mingw32/sysroot/usr/lib"; do
        if [ -f "$(cygpath -u "$potential_path")/libws2_32.a" ]; then
            platlibs=$(cygpath -u "$potential_path")
            break
        fi
    done

    if [ -f "$platlibs/libws2_32.a" ]; then
        export LDFLAGS="$LDFLAGS -L$platlibs"
    else
        echo "Warning: Could not find libws2_32.a"
    fi
    export DLLTOOL=x86_64-w64-mingw32-dlltool.exe
    export NM=x86_64-w64-mingw32-nm.exe
    export OBJDUMP=x86_64-w64-mingw32-objdump.exe

    # Check for winpthread in standard locations
    for lib in libwinpthread libpthread_win32 libpthread; do
        if [ -f "$BUILD_PREFIX_M/Library/lib/${lib}.a" ] || [ -f "$BUILD_PREFIX_M/Library/lib/${lib}.dll.a" ]; then
        export PTHREAD_LIBS="-l${lib#lib}"
        break
        fi
    done
else
    # for other platforms we just need to reconf to get the correct achitecture
    echo libtoolize
    libtoolize
    echo aclocal -I $PREFIX/share/aclocal -I $BUILD_PREFIX/share/aclocal
    aclocal -I $PREFIX/share/aclocal -I $BUILD_PREFIX/share/aclocal
    echo autoconf
    autoconf
    echo automake --force-missing --add-missing --include-deps
    automake --force-missing --add-missing --include-deps

    export CONFIG_FLAGS="--build=${BUILD}"
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

./configure "${configure_args[@]}"
make -j$CPU_COUNT
make install

# Disable make check because it fails on win-64:
# FAIL: Array.exe
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" != "1" && "${do_check:-yes}" == "yes" ]]; then
    make check
fi

rm -rf $uprefix/share/doc/libXdmcp
