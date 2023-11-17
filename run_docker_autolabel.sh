 #!/bin/bash

gpu=$1

docker run --rm -it --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) \
  -v /home/ubuntu/Loic/openpcdet_autolabel/data/:/home/data \
  -v /mnt/Autolabelling/converted_to_kitti_format/waymo_in_kitti_format/training/velodyne/:/home/data/autolabel_waymo_old/training/velodyne/ \
  -v /mnt/Autolabelling/converted_to_kitti_format/waymo_in_kitti_format/training/velodyne/:/home/data/autolabel_waymo/training/velodyne/ \
  -v /mnt/Autolabelling/converted_to_kitti_format/waymo_in_kitti_format/training/velodyne/:/home/data/waymo_og/training/velodyne/ \
  -v /mnt/Autolabelling/converted_to_kitti_format/waymo_in_kitti_format/training/label_kitti_carla_all:/home/data/waymo_all \
  -v /home/ubuntu/Loic/openpcdet_autolabel/convert_datasets:/home/convert_datasets \
  -v /mnt/Autolabelling/KITTI/training:/home/data/autolabel_og/training \
  -v /mnt/Autolabelling/KITTI/training/velodyne:/home/data/autolabel_retrain/training/velodyne \
  -v /mnt/Autolabelling/KITTI/training/velodyne:/home/data/autolabel_transfer/training/velodyne \
  -v /mnt/Autolabelling/KITTI/training/velodyne:/home/data/autolabel_show/training/velodyne \
  -v /mnt/Autolabelling/KITTI/training/velodyne:/home/data/autolabel_kitti/training/velodyne \
  -v /mnt/Autolabelling/edgar_dataset/rosbag2_2023_09_29-14_40_51_pcd/pcd:/home/data/autolabel_EDGAR/training/velodyne \
  -v /home/ubuntu/Loic/trained_models/models_kitti:/home/output \
  -v /home/ubuntu/Loic/openpcdet_autolabel/tools/cfgs/autolabel_models:/home/tools/cfgs/autolabel_models \
  -v /home/ubuntu/Loic/openpcdet_autolabel/autolabel_data/:/home/autolabel_data \
  -v /home/ubuntu/Loic/openpcdet_autolabel/autolabel_pipeline/:/home/autolabel_pipeline \
  openpcdet:autolabel bash