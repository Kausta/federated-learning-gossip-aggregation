#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_"$1".pt --peers peer-circular.txt &>../out200-circular-"$1".txt
