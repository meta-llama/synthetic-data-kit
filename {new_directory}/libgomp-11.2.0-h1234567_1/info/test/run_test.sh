

set -ex



test -f ${PREFIX}/lib/libgomp.so.1.0.0
test ! -f ${PREFIX}/lib/libgomp.so.1
exit 0
