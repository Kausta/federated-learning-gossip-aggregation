#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_"$1".pt --decay-rate 0.01 &>../out200-decay-"$1".txt
