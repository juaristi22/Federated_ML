#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

for i in $(seq 1 128); do
    for j in $(seq 1 10); do
        echo "Running $i clients with a branching factor of $j"
        python Hierarchy_Experiments.py --max_n_models $i --max_bf $j &
    done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
