#!/bin/bash


MASTER_IP=128.196.142.55
MASTER_PORT=9123
TIME_OUT_TIME=1800

MAX_WORKER=$(nproc)
MAX_WORKER=$(expr $MAX_WORKER - 1)

echo number of worker: $MAX_WORKER
echo singularity version: $(singularity --version)
echo cctools version: $(work_queue_worker --version)

for ((i = 0 ; i < $MAX_WORKER; i++)); do
	work_queue_worker $MASTER_IP $MASTER_PORT -b 30 --cores=1 -t $TIME_OUT_TIME --workdir=/scratch  &
	echo $! >> wq_pid.txt
done


