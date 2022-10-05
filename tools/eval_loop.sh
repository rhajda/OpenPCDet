#!/bin/bash

NETWORK=pointrcnn # pointrcnn or pointpillar
EVAL_TAG=sim_no_noise # real or sim_no_noise or sim_noise002 or sim_noise_down
                        # EVAL_TAG = test dataset (same as mounted dataset in docker)

# TRAIN_DATASET = train dataset
for TRAIN_DATASET in default_real_run sim_no_noise sim_noise002 sim_noise_down
do
    # i = 5 training repetitions for each training dataset
    for i in {1..5}
        do
                python test.py --cfg_file cfgs/indy_models/${NETWORK}.yaml --extra_tag "${TRAIN_DATASET}_${i}" --ckpt_dir /home/output/indy_models/${NETWORK}/${TRAIN_DATASET}_${i}/ckpt/ --eval_all --start_epoch 46 --eval_tag ${EVAL_TAG} --max_waiting_mins 0
        done
done