#/bin/bash

# set up ld specific symbolic links

echo "install ld ..."

set -ex

cd build

CHOST="${ctng_triplet}"
OHOST="${ctng_triplet_old}"

mkdir -p $PREFIX/bin
mkdir -p $PREFIX/$OHOST/bin
mkdir -p $PREFIX/$CHOST/bin

if [[ $target_platform == osx-* ]]; then
  echo "no ld support ..."
else
  cp $PWD/prefix_strip/bin/$CHOST-ld $PREFIX/bin/$CHOST-ld
  if [[ "${CHOST}" != "${OHOST}" ]]; then
    ln -s $PREFIX/bin/$CHOST-ld $PREFIX/bin/$OHOST-ld
    ln -s $PREFIX/bin/$CHOST-ld $PREFIX/$OHOST/bin/ld
  fi
  ln -s $PREFIX/bin/$CHOST-ld $PREFIX/$CHOST/bin/ld
fi

echo "ld installed"

