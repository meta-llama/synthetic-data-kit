

set -ex



test -f $PREFIX/include/X11/Xdmcp.h
test -f $PREFIX/lib/libXdmcp.so
exit 0
