set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
declare -a COMMON_MAKE_OPTS=()
COMMON_MAKE_OPTS+=(prefix=${PREFIX} exec_prefix=${PREFIX})

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/gcc-final/

make -C $CHOST/libstdc++-v3/src "${COMMON_MAKE_OPTS[@]}" install
make -C $CHOST/libstdc++-v3/include "${COMMON_MAKE_OPTS[@]}" install
make -C $CHOST/libstdc++-v3/libsupc++ "${COMMON_MAKE_OPTS[@]}" install

popd

