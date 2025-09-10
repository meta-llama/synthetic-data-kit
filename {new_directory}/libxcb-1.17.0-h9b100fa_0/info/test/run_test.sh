

set -ex



test -f $PREFIX/lib/libxcb.so
test -f $PREFIX/lib/libxcb-composite.so
test -f $PREFIX/lib/libxcb-damage.so
test -f $PREFIX/lib/libxcb-dpms.so
test -f $PREFIX/lib/libxcb-dri2.so
test -f $PREFIX/lib/libxcb-glx.so
test -f $PREFIX/lib/libxcb-present.so
test -f $PREFIX/lib/libxcb-randr.so
test -f $PREFIX/lib/libxcb-record.so
test -f $PREFIX/lib/libxcb-res.so
test -f $PREFIX/lib/libxcb-screensaver.so
test -f $PREFIX/lib/libxcb-shape.so
test -f $PREFIX/lib/libxcb-shm.so
test -f $PREFIX/lib/libxcb-sync.so
test -f $PREFIX/lib/libxcb-xf86dri.so
test -f $PREFIX/lib/libxcb-xfixes.so
test -f $PREFIX/lib/libxcb-xinerama.so
test -f $PREFIX/lib/libxcb-xkb.so
test -f $PREFIX/lib/libxcb-xtest.so
test -f $PREFIX/lib/libxcb-xv.so
test -f $PREFIX/lib/libxcb-xvmc.so
test -f $PREFIX/lib/libxcb-dri3.so
test -f $PREFIX/lib/libxcb-render.so
test -f $PREFIX/lib/libxcb-xinput.so
exit 0
