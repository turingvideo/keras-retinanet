#!/usr/bin/env bash

keras_retinanet/bin/train.py \
    --freeze-backbone --random-transform \
    --weights ../data/via-samples-10/resnet50_coco_best_v2.1.0.h5 \
    --batch-size 8 --steps 10 --epochs 10 \
    --gpu 0,1 \
    --multi-gpu 2 \
    --multi-gpu-force \
    via \
    '../data/via-samples-10/train_catalog' \
    '../data/via-samples-10/val_catalog'
