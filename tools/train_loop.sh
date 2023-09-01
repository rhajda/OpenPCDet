#!/bin/bash

for NETWORK in pointrcnn pointpillar
do
  for TRAIN_DATASET in real sim real2sim sim2real
  do
    for RUN in 1 2 3
    do
      python train.py --cfg_file cfgs/indy_models/${NETWORK}.yaml --extra_tag "${TRAIN_DATASET}_${RUN}" --dataset "${TRAIN_DATASET}" --workers 16
    done
  done
done