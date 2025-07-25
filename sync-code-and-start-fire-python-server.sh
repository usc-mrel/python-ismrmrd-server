#!/bin/bash

# Bash script to sync code and start Python ISMRMRD server
#
# First argument is path to log file.  If no argument is provided,
# logging is done to stdout (and discarded)


# Set Python's default temp folder to one that's shared with the host so that
# it's less likely to accidentally fill up the chroot
export TMPDIR=/tmp/share

cp -R -f /tmp/share/code/* "/opt/code/python-ismrmrd-server/"
mkdir -p "/opt/code/python-ismrmrd-server/seq_meta/"
cp -R -f /tmp/share/trajectories/* "/opt/code/python-ismrmrd-server/seq_meta/"
cp -R -f /tmp/share/configs/* "/opt/code/python-ismrmrd-server/configs/"

if [ $# -eq 2 ]; then
  LOG_FILE=${1}
  python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=${2} -l=${LOG_FILE} &
else
  python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=${1} &
fi

