 #!/bin/bash
gpu=$1
docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /mnt/Autolabelling/KITTI:/home/data/autolabel -v /home/ubuntu/Loic/trained_models/models_kitti:/home/output -v /home/ubuntu/Loic/openpcdet_autolabel/autolabel_pipeline/:/home/autolabel_pipeline openpcdet:autolabel bash
