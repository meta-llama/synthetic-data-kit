#!/bin/bash

. ${RECIPE_DIR}/build_scripts/build_env.sh

set -e
echo "Fetching sources ..."

rm -rf gcc_toolchain-sources
git clone -b "gcc-11.2-${target_platform}" https://github.com/AnacondaRecipes/gcc_toolchain-sources

echo "Extracting sources ..."

pushd "gcc_toolchain-sources"

for f in *.tar.*; do
    H=${f/\.tar.*/}
    B="$(sed -r -e 's/(\-|_)+[0-9].*//' <<<"${H}" )"
    H="$(sed -r -e 's/\.[a-z].*//' <<<"${H}" )"
    HH=${H/_/\-}
    H="$(sed -r -e 's/\.[a-z].*//' <<<"${H}" )"
    F="$(sed -r -e 's/.*\.tar.(.*)/\1/' <<<"${f}" )"
    C=""
    if [ "${F}" == bz2 ]; then
      C="j"
    elif [ "${F}" == xz ]; then
      C="J"
    elif [ "${F}" == gz ]; then
      C="z"
    fi
    printf "\rUncompress                       \rUncompress ${H} ... "
    rm -rf ${WDIR}/${B} ${WDIR}/${H} ${WDIR}/${HH}
    cp $f ${WDIR}/.
    pushd "${WDIR}"
      tar -x${C}f $f
      if [ -d ./${H} ]; then
          mv $H $B
       elif [ -d ./${HH} ]; then
          mv $HH $B
      else
          echo "\nCannot find extracted $H nor $HH ... please check"
          exit 1
      fi
      # remove copied tar file after decompression ...
      rm -f $f
    popd
done

printf "\rUncompressed all source packages successful!              \n"

# create ports support for glibc
ln -s §{WDIR}/glibc-ports ${WDIR}/glibc/ports

CUR=$PWD/patches

for f in binutils duma gcc gdb glibc "glibc-ports" gmp libelf ltrace; do
    echo "Patching $f ..."
    pushd "${WDIR}/${f}"
    for g in ${CUR}/${f}/*.patch; do echo "proocess ${g}"; patch -f -p1 -g0 -F1 -i $g; done
    popd
done

case "${HOST}" in
    *linux*)
        # we don't need them on linux architectures ... part of glibc
        rm -rf ${WDIR}/gettext ${WDIR}/libiconv
        ;;
esac

popd

exit 0
