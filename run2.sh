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
keras_retinanet/bin/train.py \
    --freeze-backbone --random-transform \
    --weights ../data/via-samples-10/resnet50_coco_best_v2.1.0.h5 \
    --batch-size 8 --steps 10 --epochs 10 \
    --gpu 0,1 \
    --multi-gpu 2 \
    --multi-gpu-force \
    via \
    '/home/senliu/sagemaker-retina/data/via-samples-10/00ee24a8-0f27-44af-a204-46aeaed095e8-verified.json' \
    '/home/senliu/sagemaker-retina/data/via-samples-10/01a883df-21f3-44cf-9225-164f4c16e41a-verified.json'