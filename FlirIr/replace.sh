#!/bin/bash

# Changes to current pwd
# This script assumes PIPE_PATH is line number 6
# Change number accordingly if required

DATE="${1%/}"

echo "DATE='${DATE}'" >> run.sh.s11ALL
sed -i '4d' run.sh.s11ALL
sed -ni '4{h; :a; n; ${p;x;bb}; H; ba}; :b; p' run.sh.s11ALL
