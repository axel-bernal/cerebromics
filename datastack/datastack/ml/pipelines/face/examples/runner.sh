#!/bin/bash

AUTOMATOR=$(python -c 'import datastack, os; print os.path.dirname(datastack.__file__)')
AUTOMATOR=$AUTOMATOR"/ml/pipelines/face/automator.py"

thisdir=$(pwd)

for path in ./*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"
    echo "Running "$dirname
    cd $dirname
    python $AUTOMATOR > run.log 2>&1
    cd $thisdir
done
