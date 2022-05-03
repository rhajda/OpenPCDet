#!/bin/bash

docker run --rm -it --gpus '"device=1"' --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /home/calcstudenten/zulfaj_huch/data/indy_real:/home/data/indy openpcdet:indy bash

