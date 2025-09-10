#/bin/bash

# contains CHOST binaries, and OHOST binaries as links to CHOST one
# HOST/bin/tools point to /bin/CHOST-tools
# OHOST/bin/tools point to /bin/CHOST-tools
set -x

echo "install binutils ..."

# make sure build dependencies are specific to archs
LDDEPS="as dwp gprof ld.bfd ld.gold"
if [[ "$target_platform" == osx-* ]]; then
  LDDEPS=""
fi

export CHOST="${ctng_triplet}"
export OHOST="${ctng_triplet_old}"

mkdir -p $PREFIX/$OHOST/bin

cd build

cp -r prefix_strip/* $PREFIX/.

# Remove hardlinks and replace them by softlinks
for tool in addr2line ar c++filt elfedit ${LDDEPS} nm objcopy objdump ranlib readelf size strings strip; do
  rm -rf $PREFIX/$CHOST/bin/$tool || true
  ln -s $PREFIX/bin/$CHOST-$tool $PREFIX/$CHOST/bin/$tool || true;
  if [[ "${OHOST}" != "${CHOST}" ]]; then
    ln -s $PREFIX/bin/$CHOST-$tool $PREFIX/$OHOST/bin/$tool || true;
    ln -s $PREFIX/bin/$CHOST-$tool $PREFIX/bin/$OHOST-$tool || true;
  fi
done

echo "binutils installed"
