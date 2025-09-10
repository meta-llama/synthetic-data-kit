#!/bin/bash

PERL="${BUILD_PREFIX}/bin/perl"
declare -a _CONFIG_OPTS
_CONFIG_OPTS+=(--prefix=${PREFIX})
_CONFIG_OPTS+=(--libdir=lib)
_CONFIG_OPTS+=(shared)
_CONFIG_OPTS+=(threads)
_CONFIG_OPTS+=(no-ssl2)     # broken, insecure protocol
_CONFIG_OPTS+=(no-ssl3)     # broken, insecure protocol
_CONFIG_OPTS+=(no-zlib)
_CONFIG_OPTS+=(enable-legacy) # necessary to support some function in Python package cryptography

_BASE_CC=$(basename "${CC}")
if [[ ${_BASE_CC} == *-* ]]; then
  # We are cross-compiling or using a specific compiler.
  # do not allow config to make any guesses based on uname.
  _CONFIGURATOR="perl ./Configure"
  case ${_BASE_CC} in
    x86_64-*linux*)
      _CONFIG_OPTS+=(linux-x86_64)
      CFLAGS="${CFLAGS} -Wa,--noexecstack"
      ;;
    aarch64-*-linux*)
      _CONFIG_OPTS+=(linux-aarch64)
      CFLAGS="${CFLAGS} -Wa,--noexecstack"
      ;;
    *darwin-arm64*|*arm64-*-darwin*)
      _CONFIG_OPTS+=(darwin64-arm64-cc)
      ;;
    *darwin*)
      _CONFIG_OPTS+=(darwin64-x86_64-cc)
      ;;
  esac
else
  if [[ $(uname) == Darwin ]]; then
    _CONFIG_OPTS+=(darwin64-x86_64-cc)
    _CONFIGURATOR="perl ./Configure"
  else
    # Use config, which is a config.guess-like wrapper around Configure
    _CONFIGURATOR=./config
  fi
fi

CC=${CC}" ${CPPFLAGS} ${CFLAGS}" \
  ${_CONFIGURATOR} ${_CONFIG_OPTS[@]} ${LDFLAGS}

# This is not working yet. It may be important if we want to perform a parallel build
# as enabled by openssl-1.0.2d-parallel-build.patch where the dependency info is old.
# makedepend is a tool from xorg, but it seems to be little more than a wrapper for
# '${CC} -M', so my plan is to replace it with that, or add a package for it? This
# tool uses xorg headers (and maybe libraries) which is unfortunate.
# http://stackoverflow.com/questions/6362705/replacing-makedepend-with-cc-mm
# echo "echo \$*" > "${SRC_DIR}"/makedepend
# echo "${CC} -M $(echo \"\$*\" | sed s'# --##g')" >> "${SRC_DIR}"/makedepend
# chmod +x "${SRC_DIR}"/makedepend
# PATH=${SRC_DIR}:${PATH} make -j1 depend

make -j${CPU_COUNT}

# expected error: https://github.com/openssl/openssl/issues/6953
#    OK to ignore: https://github.com/openssl/openssl/issues/6953#issuecomment-415428340
rm test/recipes/04-test_err.t

# When testing this via QEMU, even though it ends printing:
# "ALL TESTS SUCCESSFUL."
# .. it exits with a failure code.
if [[ "${HOST}" == "${BUILD}" ]]; then
  # Using verbosity on failed (sub-)tests only VF=1

  # 2025/7/22: Skip the problematic CMP HTTP test on Linux platforms for v3.0.17. Check if a new release fixed the problem.
  # It appears that the test expects the IPv6 connection to fail (return code 1) on systems
  # that don't support IPv6, but our Linux systems have IPv6 support enabled by default
  # with the loopback interface properly supporting ::1.
  # The test is connecting to http://[::1]:42871/pkix/ and expects this to fail (return code 1),
  # but it's succeeding (return code 0) on Linux systems.
  # On macOS and Windows, IPv6 loopback connectivity may be configured differently or disabled by default.
  #
  # Example from the logs:
  # None INFO # cmp_main:apps/cmp.c:2832:CMP info: using section(s) 'Mock connection' of OpenSSL configuration file '../Mock/test.cnf'
  # None INFO # opt_str:apps/cmp.c:2316:CMP warning: -proxy option argument is empty string, resetting option
  # None INFO # setup_client_ctx:apps/cmp.c:2009:CMP info: will contact http://[::1]:42871/pkix/
  # None INFO # CMP info: sending IR
  # None INFO # CMP error: connect timeout
  # None INFO # CMP error: transfer error:request sent: IR, expected response: IP
  # None INFO ../../../../util/wrap.pl ../../../../apps/openssl cmp -config ../Mock/test.cnf -section 'Mock connection' -certout 
  # ../../../../test-runs/test_cmp_http/test.cert.pem -proxy '' -no_proxy 127.0.0.1 -server '[::1]:42871' => 1
  # None INFO     not ok 3 - disabled as not supported by some host IP configurations: server IPv6 address
  if [[ "${HOST}" =~ .*linux.* ]]; then
    make test TESTS='-test_cmp_http*' V=1 > testsuite.log 2>&1 || true
  else
    make test V=1 > testsuite.log 2>&1 || true
  fi

  if ! cat testsuite.log | grep -i "all tests successful"; then
    echo "Testsuite failed!  See $(pwd)/testsuite.log for more info."
    cat $(pwd)/testsuite.log
    exit 1
  fi
fi
make install_sw install_ssldirs

# https://github.com/ContinuumIO/anaconda-issues/issues/6424
if [[ ${HOST} =~ .*linux.* ]]; then
  if execstack -q "${PREFIX}"/lib/libcrypto.so.3.0 | grep -e '^X '; then
    echo "Error, executable stack found in libcrypto.so.3.0"
    exit 1
  fi
fi
