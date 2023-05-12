 #!/bin/bash
gpu=$1
docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /mnt/EDGAR/03_data/00_Real/20221207_munich_dataset_v1:/home/Edgar -v /mnt/KITTI:/home/Kitti openpcdet:sim2real bash

