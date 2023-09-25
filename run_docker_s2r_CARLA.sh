#!/bin/bash

gpu=$1
docker run --rm -it --name "CARLA" --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /mount/huch/CARLA_KITTI_2/CARLA:/home/data/s2r -v /data/salinas_huch/sim2real_trainings/train_CARLA:/home/output openpcdet:sim2real bash

