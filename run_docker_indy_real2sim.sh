#!/bin/bash

gpu=$1
docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /home/ubuntu/Huch/data/Sim2RealDistributionAlignedDataset/real2sim:/home/data/indy -v /home/ubuntu/Huch/OpenPCDet_output:/home/output openpcdet:indy113 bash

