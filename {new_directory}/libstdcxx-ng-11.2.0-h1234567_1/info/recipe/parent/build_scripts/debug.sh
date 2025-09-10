#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

bash ${RECIPE_DIR}/build_scripts/duma_target.sh
[ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/ltrace_target.sh
[ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/strace_target.sh
[ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gdb_server.sh
[ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gdb_target.sh
[ $? -eq 0 ]

exit 0
