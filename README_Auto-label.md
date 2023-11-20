
------------------------------------------------------------------------------------------------------------------------
                            -- This is the README file of the Auto-label pipeline.
                                Developed within the scope of a Master's Thesis.  --

                            Developer:
                            Loïc Stratil
                            @ Chair of Automotive Technology, Technical University of Munich

------------------------------------------------------------------------------------------------------------------------

Abbreviations:

autom. : automatically
PL : Pseudo-labels
BBOX : Bounding Box
GT : Ground Truth

------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------
-- Code structure - (Auto-label) Data --
----------------------------------------------------------------

|-- /openpcdet_autolabel

    |-- /autolabel_data
        |-- /autolabel_XXX                              --- Contains all data generated during an Auto-label iteration.
            |-- /models                                 --- Selected object detectors used in the Auto-label ensemble.
                |-- /past_iterations
                    |-- /iteration_XX
                |-- /pointpillar                        --- The selected PointPillars epoch is autom. loaded here.
                    |-- checkpoint_epoch_XX.pth
                |-- /pointrcnn                          --- The selected PointRCNN epoch is autom. loaded here.
                    |-- checkpoint_epoch_XX.pth
                |-- /second                             --- The selected SECOND epoch is autom. loaded here.
                    |-- checkpoint_epoch_XX.pth

            |-- /predictions                            --- Contains all proposals and pseudo-labels of an Auto-label iteration.
                |-- /pseudo_labels
                    |-- /majority_voting                --- Voted pseudo-labels if majority voting is used.
                        |-- XXX.txt
                    |-- /nms_voting                     --- Voted pseudo-labels if nms voting is used.
                        |-- XXX.txt
                |-- /pointpillar                        --- PointPillars proposals.
                    |-- XXX.csv
                |-- /pointrcnn                          --- PointRCNN proposals.
                    |-- XXX.csv
                |-- /second                             --- SECOND proposals.
                    |-- XXX.csv

            |-- /output_labels                          --- Pseudo-labels in output format. Used to update /data/..label_2
                    |-- /majority_voting                --- If majority voting is used
                        |-- XXX.txt
                    |-- /nms_voting                     --- If nms voting is used
                        |-- XXX.txt

    |-- /data                                           --- Folder contains initial unlabeled point clouds. LiDAR-based format.
        |-- /autolabel_XXX                              --- files in LiDAR-based format (only .pcds necessary)
            |-- /ImageSets_KITTI_full
                |-- train.txt
                |-- pseudo-label-txt
                |-- test.txt
                |-- val.txt

            |-- /original                               --- Backup of initial data. (In case of labeled data use.)

            |-- /training
                |-- /label_2                            ---> Output pseudo-labels are stored here.
                    |-- XXX.txt
                |-- /velodyne                           --- .pcd point clouds.
                    |-- XXX.pcd

----------------------------------------------------------------
-- Code structure - Auto-label pipeline --
----------------------------------------------------------------

|-- /openpcdet_autolabel

    |-- /autolabel_pipeline                             --- Contains all scripts for Auto-label pipeline.
        |-- /Helperfunctions
            |--                                         --- Old / debugging scripts. Not relevant for Auto-label pipeline.

        |-- /SemiSupervisedLearning                     --- Secondary scripts for Dataset preparation.
            |-- ImageSet_from_files_in_folder.py        --- create ImageSet .txt files from the /label_2 folder.
            |-- reduce_waymo_dataset_size.py            --- Reduces the number of dataset files to a predefined number.
            |-- split_train_pseudolabel_KITTI.py        --- Split og train.txt to train.txt + pseudo_label.txt with split ratio.

        |-- /voting_schemes
            |-- majority_voting.py                      --- Majority voting function used within Auto-label pipeline.
            |-- nms_voting.py                           --- NMS voting function used within Auto-label pipeline.

        |-- LiDAR_based_format_from_pcds.py             --- Creates the /data/autolabel_XXX format from .pcd files as input.
        |-- base_functions.py                           --- Contains config loader and path manager for Auto-label pipeline.
        |-- autolabel.yaml                              --- Central config file for the Auto-label process.
        |-- predict_objects.py                          --- Function generates pseudo-label proposals for all 3 base learners.
        |-- predict_objects.sh                          --- .sh file for proposal generation of a single detector. (for debugging)
        |-- main_autolabel.py                           --- Function that votes pseudo-labels taking folders with proposals as input.

        |-- main_pipeline.py                            ---> Main Auto-label pipeline trigger function (including modes).
        |-- visualize_pcds.py                           --- PCD and BBOX visualizer (trigger directly). Configurable via autolabel.yaml.

        |-- evaluate_labels.py                          --- Outputs performance metrics for the pseudo-labeled data (Code analogous to pcdet + added bounding box dimension evaluation).
        |-- sensitivity_analysis_thresholds.py          --- Takes PL-proposals and votes PLs for a grid of parameters to enable sensitivity analysis.
        |-- evaluate_GT_dimensions.py                   --- Get class-wise BBOX info regarding dimensions + 2D distribution scatter plots

    |-- /pcdet                                          --- pcdet library (https://github.com/open-mmlab/OpenPCDet)

    |-- /tools                                          --- pcdet library (https://github.com/open-mmlab/OpenPCDet)
        |-- /autolabel_models
            |-- pointpillar.yaml                        --- PointPillars model config for pcdet. Parameters are adapted in main_pipeline.py for Auto-label pipeline.
            |-- pointrcnn.yaml                          --- PointRCNN model config for pcdet. Parameters are adapted in main_pipeline.py for Auto-label pipeline.
            |-- second.yaml                             --- SECOND model config for pcdet. Parameters are adapted in main_pipeline.py for Auto-label pipeline.

        |-- /dataset_configs
            |-- autolabel_dataset.yaml                  --- Dataset config for pcdet. Parameters are adapted in main_pipeline.py for Auto-label pipeline.

---------------------------------------------------------------
-- Code structure - Auto-label Docker --
----------------------------------------------------------------

|-- /openpcdet_autolabel

    |-- /docker                                         --- Contains the base docker with CUDA 11.3 (openpcdet-base113).
        |-- Dockerfile                                  --- Dockerfile for openpcdet-base113 image (base image).
        |-- build_docker.sh                             --- Build openpcdet-base113 image.

    |-- build_docker_autolabel.sh                       --- Build auto-label image (Multi-stage, takes openpcdet-base113 as base).
    |-- Dockerfile                                      --- Dockerfile for auto-label image.
    |-- run_docker_autolabel.sh                         --- Opens a docker container for Auto-label pipeline. Dataset data should be mounted here.

---------------------------------------------------------------
-- Code structure - Dataset converter --
----------------------------------------------------------------

|-- /openpcdet_autolabel

    |-- /convert_datasets                               --- Contains waymo and (nuscenes) data converters to KITTI camera format.
        |-- /nuscenes                                   --- nuscenes to KITTI camera format.
            |-- convert_nuscenes_to_kitti.py

        |-- /waymo_to_kitti_converter                   --- waymo to KITTI camera format.
            |-- check_labels.py                         --- Checks the number of GT labels in two separate folders.
            |-- convert_waymo_to_kitti.py               --- Convert waymo data to KITTI camera format.
            |-- convert_kitti_cam_to_kitti_carla.py     --- Convert KITTI camera format to LiDAR-based format (Auto-label pipeline format)
            |-- rename_kitti_carla_dataset.py           --- Convert Waymo to LiDAR-based nomenclature
            |-- waymo_to_kitti_carla_transform.py       --- convert_waymo_to_kitti.py + convert_kitti_cam_to_kitti_carla.py. Rename is not included.

        |-- Dockerfile                                  --- Dockerfile for converter_docker image.
        |-- build_docker_converter.sh                   --- Build converter_docker image.
        |-- run_docker_converter.sh                     --- Opens a converter_docker docker container. Dataset data should be mounted here.

---------------------------------------------------------------
-- Code structure - Original PCDET backups -- (not relevant for Auto-label pipeline)
----------------------------------------------------------------
|-- /openpcdet_autolabel

    |-- /docs (not relevant)
    |-- /og_git (not relevant)

------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------
-- Auto-label Pipeline - How to --
-------------------------------------------------------------------

1) Provide the data to be pseudo-labeled in LiDAR-based format.
    To run Auto-label pipeline, the data to be pseudo-labeled must be present in LiDAR-based format.
    Two initial set-ups are possible:
    - Only .pcd files are present, all contained within a specific folder. Then:
        - Trigger: ../autolabel_pipeline/LiDAR_based_format_from_pcds.py
            This script generates all necessary files and folders within the /data/autolabel_XXX project.
            Prerequisite to this script: The .pcd files must be placed in /data/autolabel_XXX/training/velodyne/ before trigger.
            The script then maps original .pcd names to 6 digit format and saves the mapping file.
            It renames all .pcd files according to the mapping.
            Generates /label_2 files with a dummy label inside (will be overwritten within Auto-label pipeline).
            Additionally generates the train.txt and pseudo-label ImageSets to contain all data, and initializes an empty
            val.txt file.
            The output structure is ready for use with Auto-label pipeline.
    - Manually arrange the data to fit the LiDAR-based format. The structure must be in line with /data/autolabel_XXX

2) Provide the labeled data for the initial Auto-labeling iteration in LiDAR-based format.
    Within this work, the KITTI and Waymo Open Datasets have been utilized.
    The Docker image in "Code structure - Dataset converter" can directly be built and corresponding scripts used.
    - KITTI: Labeled bounding boxes must be present in LiDAR-based format. Conversion from KITTI camera format to
        LiDAR based format can be made with convert_kitti_cam_to_kitti_carla.py. If necessary, rename the label and .pcd files
        with rename_kitti_carla_dataset.py.
    - Waymo: Waymo dataset can be converted to KITTI camera format with convert_waymo_to_kitti.py
        (Sampling rate is fixed to 10Hz but can be changed within the code.). After running this script run convert_kitti_cam_to_kitti_carla.py
        this transforms the Waymo data in KITTI camera format to LiDAR based format. Finally trigger rename_kitti_carla_dataset.py
        to rename the .pcd files and labels to 6 digit format.
    - Nuscenes: A nuscenes converter is present within /convert_datasets/nuscenes but has not been tested in this work.
        After converting nuscenes labels to KITTI camera format, use convert_kitti_cam_to_kitti_carla.py and rename_kitti_carla_dataset.py
        in analogy to waymo. Debugging could be necessary.
    - Other datasets: convert_kitti_cam_to_kitti_carla.py and rename_kitti_carla_dataset.py can be utilized when the initial
        data is present in KITTI camera format. Converters to KITTI camera format have been proposed for various public datasets.

3) Create an autolabel_data project.
    - in /autolabel_data create a folder with the name of the project: autolabel_XXX
    - Inside the project create an empty /models folder.
    structure: /autolabel_data/autolabel_XXX/models (the initial trained base learner models will be placed in this folder by the
    auto-label pipeline)

4) Run the initial Auto-label iteration. Trigger main_pipeline.py with following settings:
   - Configure autolabel.yaml and main_pipeline.py.
        - In autolabel.yaml define:
            DATA.PATH_POINT_CLOUDS:
            DATA.PATH_GROUND_TRUTHS:
            PROJECT.DATASET:
            PROJECT.AUTOLABEL_DATA:

        - In main_pipeline.py configure as follows for initial iteration:
            opt = {
                    'FLAG_PREPARE_MODELS_FOLDER': False,        --- False, as no pretrained models exist at this point.
                    'MODE_INITIAL_TRAIN': True,                 --- True, as the initial iteration mode is triggered
                    '360_DEGREE_PSEUDO_LABELS': True,           --- True, if GT labels exist in 360° around the Vehicle.
                    'RESTRICT_TO_KITTI_FOV': False,             --- False, only used to compare pseudo-labels to KITTI GT
                    'EVAL_ON_KITTI_VAL': False                  --- False, no evaluation.
                  }

            modules = {
                    'FLAG_RESET_PSEUDO_LABEL_FOLDERS': False,   --- False, no pretrained models exist.
                    'FLAG_PREDICT_OBJECTS': False,              --- False, initial training only. No Auto-label iteration.
                    'FLAG_VOTE_PSEUDO_LABELS': False,           --- False, initial training only. No Auto-label iteration.
                    'FLAG_COMPUTE_EVALUATION_METRICS': False,   --- False, initial training only. No evaluation.
                    'FLAG_CONVERT_PSEUDO_LABELS': False,        --- False, initial training only. No Auto-label iteration.
                    'FLAG_BACKUP_OG_TRAIN': False,              --- False, initial training only. No Auto-label iteration.
                    'FLAG_UPDATE_TRAINSET': False,              --- False, initial training only. No Auto-label iteration.
                    'FLAG_CREATE_AUTOLABEL_INFOS': True,        --- True, necessary for pcdet training.
                    'FLAG_TRAIN': True                          --- True, initial training.
                      }
   - Run main_pipeline.py from auto-label image container (run_docker_autolabel.sh, make sure the dataset data is mounted here).

5) Run an Auto-label iteration. Trigger main_pipeline.py with following settings:
    - Configure autolabel.yaml and main_pipeline.py.
        - In autolabel.yaml define:
                PIPELINE.VOTING_SCHEME: --> "NMS" or "MAJORITY"
                Specify all MAJORITY_VOTING or NMS_VOTING parameters

        - In main_pipeline.py configure as follows for initial iteration:
            opt = {
                    'FLAG_PREPARE_MODELS_FOLDER': True,         --- Run ONCE per iteration, moves selected epochs to /autolabel_data/autolabel_XXX/models
                    'MODE_INITIAL_TRAIN': False,                --- False, as the auto-label iteration is triggered
                    '360_DEGREE_PSEUDO_LABELS': True,           --- True, if GT labels exist in 360° around the Vehicle.
                    'RESTRICT_TO_KITTI_FOV': False,             --- False, only used to compare pseudo-labels to KITTI GT.
                    'EVAL_ON_KITTI_VAL': False                  --- True, only if GT data exists to compare PLs against.
                  }

            modules = {
                    'FLAG_RESET_PSEUDO_LABEL_FOLDERS': True,    --- True, resets the /autolabel_XXX/ subfolders where necessary.
                    'FLAG_PREDICT_OBJECTS': True,               --- True, generates PL proposals and saves them to /autolabel_XXX/predictions/..
                    'FLAG_VOTE_PSEUDO_LABELS': True,            --- True, votes PLs from proposals and saves them to /autolabel_XXX/predictions/pseudo_labels/..
                    'FLAG_COMPUTE_EVALUATION_METRICS': False,   --- True, only if GT data exists to compare PLs against.
                    'FLAG_CONVERT_PSEUDO_LABELS': True,         --- True, converts PLs to correct format for re-training.
                    'FLAG_BACKUP_OG_TRAIN': True,               --- True, backup of data present in initial iteration.
                    'FLAG_UPDATE_TRAINSET': True,               --- True, updates the /data/autolabel_XXX/ train files for re-training and output.
                    'FLAG_CREATE_AUTOLABEL_INFOS': True,        --- True, necessary for pcdet training.
                    'FLAG_TRAIN': True                          --- True, pcdet training loop.

6) Repeat Auto-label iterations. Adapt Voting parameters in autolabel.yaml according to employed strategy.
7) Visualize .pcds, pseudo-label proposals as well as voted pseudo-labels with visualize_pcds.py.
