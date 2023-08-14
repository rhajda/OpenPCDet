#!/bin/bash

for NETWORK in pointrcnn pointpillar
do
  for TRAIN_DATASET in real sim sim2real real2sim
  do
      for TEST_DATASET in real sim sim2real real2sim
          do
                  python test.py --cfg_file cfgs/indy_models/${NETWORK}.yaml --extra_tag "${TRAIN_DATASET}" --ckpt_dir /home/output/indy_models/${NETWORK}/${TRAIN_DATASET}/ckpt/ --eval_all --start_epoch 75 --dataset ${TEST_DATASET} --max_waiting_mins 0
          done
  done
done