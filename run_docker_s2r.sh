#!/bin/bash

gpu=$1
docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /home/calcstudenten/salinas/data/for_training/real:/home/data/s2r -v /data/salinas/openpcdet_trainings/sim-to-real:/home/output openpcdet:sim2real bash

