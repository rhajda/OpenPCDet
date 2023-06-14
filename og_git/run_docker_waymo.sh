#!/bin/bash

docker run --rm -it --gpus '"device=1"' --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname)  -v /home/calcmitarbeiter/huch/_Perception_Datasets/Waymo_Perception/:/home/data/waymo -v /data/huch/openpcdet_trainings/:/home/output openpcdet:latest bash
