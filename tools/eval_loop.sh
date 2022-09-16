#!/bin/bash

# DATASET = train dataset
for DATASET in default_real_run sim_no_noise sim_noise002 sim_noise_down
do
    # i = 5 training repetitions for each training dataset
    for i in {1..5}
        do
                # choose pointrcnn or pointpillar for cfg_file and ckpt_dir
                # eval_tag = test dataset (same as mounted dataset in docker)
                python test.py --cfg_file cfgs/indy_models/pointrcnn.yaml --extra_tag "${DATASET}_${i}" --ckpt_dir /home/output/indy_models/pointrcnn/${DATASET}_${i}/ckpt/ --eval_all --start_epoch 46 --eval_tag sim_noise_down --max_waiting_mins 0
        done
done