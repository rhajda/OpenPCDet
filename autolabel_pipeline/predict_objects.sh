
#!/usr/bin/env bash

# Assign arguments to the variables
CFG_FILE="/home/tools/cfgs/autolabel_models/pointpillar.yaml"
CKPT_DIR="/home/output/home/tools/cfgs/autolabel_models/pointpillar/default/ckpt/"
CKPT='checkpoint_epoch_75.pth'

# Run:
python predict_objects.py --cfg_file "$CFG_FILE" --ckpt_dir "$CKPT_DIR" --ckpt ${CKPT}