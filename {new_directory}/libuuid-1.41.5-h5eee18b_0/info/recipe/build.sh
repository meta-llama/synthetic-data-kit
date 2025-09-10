#!/bin/bash
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/gnuconfig/config.* ./config

bash configure --prefix=$PREFIX --disable-all-programs --enable-libuuid --with-bashcompletiondir=$PREFIX/share

make
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
make tests
fi
make install

rm -fr $PREFIX/share
