#!/usr/bin/env bash

resultdir="/home/senliu/Results/WIDER/$(date +"%m-%d-%Y---%H-%M")"
logdir="${resultdir}/logs"
snapshotdir="${resultdir}/snapshots"

keras_retinanet/bin/train.py \
    --freeze-backbone --random-transform \
    --weights /home/senliu/Results/WIDER/09-25-2019---12-43/snapshots/resnet50_via_20.h5 \
    --batch-size 8 --steps 11437 --epochs 5 \
    --multi-gpu 1 \
    --multi-gpu-force \
    --tensorboard-dir "${logdir}" \
    --snapshot-path "${snapshotdir}" \
    via \
    '/home/senliu/Datasets/WIDER/annotation/train_via_no_filter.json' \
    '/home/senliu/Datasets/WIDER/annotation/val_via_no_filter.json'