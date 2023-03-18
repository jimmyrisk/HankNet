#!/bin/bash

start() {                   # function for starting python scripts
    echo "start $@"         # print which runner is started
    "$@" &                  # start python script in background
    pids+=("$!")            # save pids in array
}

start python "C:/Users/Mark Risk/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
    --run-id 3232 \
    --reward-type 1 \
    --lawn-num 21 \
    --hidden-size 32 \
    --hidden-num 32

start python "C:/Users/Mark Risk/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
    --run-id 1632 \
    --reward-type 1 \
    --lawn-num 21 \
    --hidden-size 16 \
    --hidden-num 32


start python "C:/Users/Mark Risk/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
    --run-id 3216 \
    --reward-type 1 \
    --lawn-num 21 \
    --hidden-size 32 \
    --hidden-num 16


start python "C:/Users/Mark Risk/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
    --run-id 1616 \
    --reward-type 1 \
    --lawn-num 21 \
    --hidden-size 16 \
    --hidden-num 16

start python "C:/Users/Mark Risk/OneDrive - Cal Poly Pomona/research (new)/HankNet/deepmower/main-ikostrikov.py" \
    --run-id 6464 \
    --reward-type 1 \
    --lawn-num 21 \
    --hidden-size 64 \
    --hidden-num 64


for pid in ${pids[*]}; do
    wait $pid
done