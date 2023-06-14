#!/bin/bash

docker run --rm -it --gpus '"device=2"' --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname)  -v /home/calcmitarbeiter/huch/_Perception_Datasets/KITTI_3D_Object_Detection/:/home/data/kitti -v /data/huch/openpcdet_trainings/:/home/output openpcdet:latest bash
