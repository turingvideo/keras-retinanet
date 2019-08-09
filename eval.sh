#!/usr/bin/env bash

keras_retinanet/bin/evaluate.py \
    --gpu 0 \
    --convert-model \
    --score-threshold 0.5 \
    --save-path ./eval_results \
    via '/media/fwang/Data1/PedestrianDataset/WIDER Person Challenge 2019/Annotations/val_via_no_filter_1000.json' \
    ./snapshots/finetuned/resnet50_via_01.h5
