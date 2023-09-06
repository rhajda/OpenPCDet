 #!/bin/bash

docker run --rm -it --network=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) \
  -v /home/ubuntu/Loic/convert_datasets/waymo_to_kitti_converter:/home/waymo_to_kitti_converter\
  -v /mnt/Autolabelling/waymo_open_dataset:/home/data/waymo_open_dataset \
  -v /mnt/Autolabelling/converted_to_kitti_format/:/home/data/converted_to_kitti_format/ \
  converter_docker:latest bash