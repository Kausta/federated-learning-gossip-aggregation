#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_"$1".pt &>../out200-gossip-"$1".txt
