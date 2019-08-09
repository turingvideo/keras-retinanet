#!/usr/bin/env bash

keras_retinanet/bin/train.py \
    --freeze-backbone --random-transform \
    --weights ./snapshots/finetuned/resnet50_via_01.h5 \
    --batch-size 8 --steps 10 --epochs 10 \
    --gpu 0 \
    via \
    '/media/fwang/Data1/PedestrianDataset/WIDER Person Challenge 2019/Annotations/train_via_no_filter.json' \
    '/media/fwang/Data1/PedestrianDataset/WIDER Person Challenge 2019/Annotations/val_via_no_filter_1000.json'
