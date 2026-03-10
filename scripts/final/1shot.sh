#!/bin/bash

#set -Eeuxo pipefail
gpu=0
shot=1
method='Frequency.Final.method'
backbone='Frequency.Final.backbone'

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

target_num_label=5
for target in ;
do
  name='pikey/'$method'/'$backbone'/share_ffilt/'$target'/'$shot'-shot/target_num_label-'$target_num_label
  CUDA_VISIBLE_DEVICES=$gpu python train.py  --FM --FE --FF --drop_prob 0.1 --target_num_label $target_num_label --share_ffilt --backbone $backbone --method $method --temp 1.0 \
  --stop_epoch $stop_epoch --modelType Student --target_set $target --name $name \
  --train_aug --warmup baseline --n_shot $shot
  CUDA_VISIBLE_DEVICES=$gpu python test.py --ffilter 'A' --backbone $backbone --method $method --name $name --dataset $target --save_epoch 399 --n_shot $shot
done