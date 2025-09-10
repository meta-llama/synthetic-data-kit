

set -ex



test -f $PREFIX/lib/libX11.so
test -f $PREFIX/lib/libX11-xcb.so
exit 0
