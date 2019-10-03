#!/usr/bin/env bash

# catalog: relative, json: relative
# keras_retinanet/bin/train.py \
#     --freeze-backbone --random-transform \
#     --weights ../data/via-samples-10/resnet50_coco_best_v2.1.0.h5 \
#     --batch-size 8 --steps 10 --epochs 10 \
#     --gpu 0,1 \
#     --multi-gpu 2 \
#     --multi-gpu-force \
#     via \
#     '../data/via-samples-10/train_catalog' \
#     '../data/via-samples-10/val_catalog'

# catalog: absolute, json: relative
# keras_retinanet/bin/train.py \
#     --freeze-backbone --random-transform \
#     --weights ../data/via-samples-10/resnet50_coco_best_v2.1.0.h5 \
#     --batch-size 8 --steps 10 --epochs 10 \
#     --gpu 0,1 \
#     --multi-gpu 2 \
#     --multi-gpu-force \
#     via \
#     '/home/senliu/sagemaker-retina/data/via-samples-10/train_catalog' \
#     '/home/senliu/sagemaker-retina/data/via-samples-10/val_catalog'

# json: absolute

resultdir="/home/senliu/Results/WIDER/$(date +"%m-%d-%Y---%H-%M")"
logdir="${resultdir}/logs"
snapshotdir="${resultdir}/snapshots"

keras_retinanet/bin/train.py \
    --freeze-backbone --random-transform \
    --weights /home/senliu/Results/WIDER/09-25-2019---12-43/snapshots/resnet50_via_20.h5 \
    --batch-size 8 --steps 11437 --epochs 5 \
    --gpu 0,1 \
    --multi-gpu 2 \
    --multi-gpu-force \
    --tensorboard-dir "${logdir}" \
    --snapshot-path "${snapshotdir}" \
    via \
    '/home/senliu/Datasets/WIDER/annotation/train_via_no_filter.json' \
    '/home/senliu/Datasets/WIDER/annotation/val_via_no_filter.json'