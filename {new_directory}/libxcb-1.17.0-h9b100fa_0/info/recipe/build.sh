#! /bin/bash

set -e
set -x
IFS=$' \t\n' # workaround for conda 4.2.13+toolchain bug

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

    # And we need to add the search path that lets libtool find the
    # msys2 stub libraries for ws2_32.
    # Find MSYS2 libraries directory using a more reliable approach
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
else
    # for other platforms we just need to reconf to get the correct achitecture
    echo libtoolize
    libtoolize
    echo aclocal -I $PREFIX/share/aclocal -I $BUILD_PREFIX/share/aclocal
    aclocal -I $PREFIX/share/aclocal -I $BUILD_PREFIX/share/aclocal
    echo autoheader
    autoheader
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
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
make check
fi

rm -rf $uprefix/share/man $uprefix/share/doc/${PKG_NAME#xorg-}

# Remove any new Libtool files we may have installed. It is intended that
# conda-build will eventually do this automatically.
find $uprefix/. -name '*.la' -delete

if [ "$(uname)" == "Linux" ]; then
    # Build a dummy libxcb-xlib. This library used to exist, but was
    # deprecated.
    # The problem is that on older systems where it still exists
    # (e.g SUSE 11), we may end up mixing the system's libxcb-xlib with the
    # libX11 and libxcb provided by conda.
    # This can cause missing symbols when trying to run a binary which links
    # to libxcb:
    #
    #    /usr/lib64/libxcb-xlib.so.0: undefined symbol: _xcb_unlock_io
    #
    # In this case _xcb_unlock_io is present on /usr/lib64/libxcb.so.1, but
    # there is no such function anymore on the version we built here.
    # Packaging a dummy library solves libxcb-xlib this problem as the only
    # libxcb-xlib user is libX11:
    #
    # https://lists.freedesktop.org/archives/xcb/2009-April/004489.html
    # https://lists.freedesktop.org/archives/xcb/2009-April/004492.html
    #
    # PLEASE NOTE that if you are facing this problem you also need to install
    # conda-forge's libx11, otherwise you will get something like:
    #
    #   /usr/lib64/libX11.so.6: undefined symbol: xcb_xlib_lock
    cd "$PREFIX/lib"
    echo ' ' | $CC --shared -Wl,-soname,libxcb-xlib.so.0 -o libxcb-xlib.so.0.0.0
    ln -s libxcb-xlib.so.0.0.0 libxcb-xlib.so.0
fi

# See https://github.com/conda-forge/libxcb-feedstock/issues/21 . For some
# reason, the osx-arm64 install ends up containing inappropriate __pycache__
# files in a site-packages subdirectory. Not at all clear why it only seems
# to happen on that one platform, but no need to be ultra-precise:
rm -rfv $PREFIX/lib/python*/site-packages/xcbgen
