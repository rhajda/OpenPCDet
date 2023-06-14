#!/bin/bash

gpu=$1
docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /mnt/Autolabelling/KITTI:/home/data/s2r -v /data/salinas_huch/sim2real_trainings/test_KITTI:/home/output openpcdet:sim2real bash

