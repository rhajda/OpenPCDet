#!/bin/bash

NETWORK=pointrcnn # pointrcnn or pointpillar
EVAL_TAG=real # real or sim
# EVAL_TAG = test dataset (same as mounted dataset in docker)

# TRAIN_DATASET = train dataset
for TRAIN_DATASET in real #sim sim2real real2sim
do
    # i = 1 training repetitions for each training dataset
    for i in {1..1}
        do
                python test.py --cfg_file cfgs/indy_models/${NETWORK}.yaml --extra_tag "${TRAIN_DATASET}" --ckpt_dir /home/output/indy_models/${NETWORK}/${TRAIN_DATASET}/ckpt/ --eval_all --start_epoch 75 --eval_tag ${EVAL_TAG} --max_waiting_mins 0
        done
done