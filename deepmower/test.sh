#!/bin/bash

start() {                   # function for starting python scripts
    echo "start $@"         # print which runner is started
    "$@" &                  # start python script in background
    pids+=("$!")            # save pids in array
}

#for i in {1..10}
#do
#start python "C:/Users/jrisk/Dropbox/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
#    --run_id $i
#done

for j in {1..10}
do

for i in {0..3}
do
start python "C:/Users/jrisk/Dropbox/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
    --run-id $j \
    --reward-type $i \
    --lawn-num 21
done

for pid in ${pids[*]}; do
    wait $pid
done

done