#!/bin/bash

set -e
set -x
set -E
SERVERPORT_IPAL="$1" ; shift
SERVERPORT_STATE="$1" ; shift
TRAINFILE_NAME="$1" ; shift

### Server
# mkfifo opens a stream, that will not close - IIDS will continue reading input
mkfifo IPAL_INPUT
mkfifo STATE_INPUT

# start ncat server
echo "Starting server on port $SERVERPORT_IPAL and $SERVERPORT_STATE for $(hostname -i)"

ncat -lk --append -p $SERVERPORT_IPAL | tee $TRAINFILE_NAME.ipal IPAL_INPUT &
ncat -lk --append -p $SERVERPORT_STATE | tee $TRAINFILE_NAME.state STATE_INPUT &

### Node exporter
./node_exporter --web.listen-address=":9102" &


### IDS
./ipal-iids $@ &


wait -n 
exit $?
