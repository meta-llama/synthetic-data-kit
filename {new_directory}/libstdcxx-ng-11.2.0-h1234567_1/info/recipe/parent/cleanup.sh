#!/bin/bash

set -e

printf "Cleaning up intermediate files ..."

for f in build gcc_built buildtools binutils gcc_toolchain-sources \
         libelf mpfr expat glibc glibc-ports gmp linux ncurses gcc \
         isl mpc zlib libiconv gettext duma ltrace strace gdb; do
    printf "\rCleaning up intermediate files ...                   \r"
    printf "Cleaning up intermediate files ... $f"
    rm -rf $f
done

printf "\rCleaning up intermediate files done                            \n"

exit 0
