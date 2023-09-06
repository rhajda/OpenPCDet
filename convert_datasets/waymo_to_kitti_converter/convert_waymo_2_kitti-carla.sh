
#!/bin/bash


ROOT_PATH = "/home/data/waymo_open_dataset/"
OUT_DIR = "/home/data/converted_to_kitti_format/"


# Run the Python script with the provided arguments
python create_data.py --root-path "$ROOT_PATH" --out-dir "$OUT_DIR" waymo