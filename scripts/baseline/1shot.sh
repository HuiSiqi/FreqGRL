#!/bin/bash

#set -Eeuxo pipefail
gpu=7
shot=1
target=cub
method='PrototypeMethod'

while getopts "g:s:t:m:" opt; do
    case $opt in
        g)
            gpu=$OPTARG
            ;;
        s)
            shot=$OPTARG
            ;;
        t)
            target=$OPTARG
            ;;
        m)
            method=$OPTARG
            ;;
        ?)
          echo "Invalid option: -$OPTARG"
          exit 1
    esac
done

stop_epoch=400
backbone=backbone

for target in cub cars places plantae;
do
  name='pikey/'$method'/'$backbone'/'$target'/'$shot'-shot/'
  CUDA_VISIBLE_DEVICES=$gpu python train.py --backbone $backbone --temp 1.0 --method $method \
  --stop_epoch $stop_epoch --modelType Student --target_set $target --name $name \
  --train_aug --warmup baseline --n_shot $shot
  CUDA_VISIBLE_DEVICES=$gpu python test.py --backbone $backbone --method $method --name $name --dataset $target --save_epoch 399 --n_shot $shot
done