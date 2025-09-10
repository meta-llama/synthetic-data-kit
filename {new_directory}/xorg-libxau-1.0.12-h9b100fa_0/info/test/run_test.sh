

set -ex



test -f $PREFIX/include/X11/Xauth.h
test -f $PREFIX/lib/libXau.so
exit 0
