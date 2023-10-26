#!/bin/bash

gpu=$1
docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /home/ubuntu/Huch/data/CARLA_KITTI:/home/data/s2r -v /home/ubuntu/Huch/OpenPCDet_output:/home/output openpcdet:sim2real bash

