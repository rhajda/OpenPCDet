#!/bin/bash

for NETWORK in pointrcnn pointpillar
do
  for TRAIN_DATASET in real sim real2sim sim2real
  do
    for RUN in 1 2 3
    do
      for TEST_DATASET in real sim real2sim sim2real
      do
        python test.py --cfg_file cfgs/indy_models/${NETWORK}.yaml --extra_tag "${TRAIN_DATASET}_${RUN}" --ckpt_dir /home/output/indy_models/${NETWORK}/${TRAIN_DATASET}_${RUN}/ckpt/ --eval_all --start_epoch 66 --dataset ${TEST_DATASET} --max_waiting_mins 0
      done
    done
  done
done