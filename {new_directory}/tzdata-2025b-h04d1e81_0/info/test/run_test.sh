#!/usr/bin/env bash
# Run the tests for tzdata

set -ex

exists() {
	FULL_PATH="${PREFIX}/${1}"
	if [ -f "${FULL_PATH}" ]; then
		echo "Found ${1}"
	else
		echo "Could not find ${FULL_PATH}"
		exit 1
	fi
}

for i in share/zoneinfo/{zone,iso3166,zone1970}.tab share/zoneinfo/leapseconds share/zoneinfo/leap-seconds.list share/zoneinfo/tzdata.zi; do
	exists $i
done

# Make sure we only package zoneinfo and nothing else.
dirs="$(find "${PREFIX}" -mindepth 1 -maxdepth 2 ! -path "${PREFIX}/share" ! -path "${PREFIX}/conda-meta*")"
test "${dirs}" = "${PREFIX}/share/zoneinfo"

# Make sure we only package timezone information files.
heads="$(
        find "${PREFIX}/share/zoneinfo" -type f \
          \! -name \*.zi \! -name \*.tab \! -name leapseconds \! -name leap-seconds.list \
          -exec sh -c 'head -c4 $1 && echo' sh {} \; \
          | uniq
      )"
test "${heads}" = TZif
