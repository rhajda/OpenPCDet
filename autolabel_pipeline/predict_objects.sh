
#!/usr/bin/env bash

# Assign arguments to the variables
# Writes the resulting predictions to /home/autolabel_pipeline/predictions/ ##model/ ##epoch/

CFG_FILE="/home/tools/cfgs/autolabel_models/second.yaml"
CKPT_DIR="/home/output/home/tools/cfgs/autolabel_models/second/default/ckpt/"
CKPT='checkpoint_epoch_80.pth'


# Run:
python predict_objects.py --cfg_file "$CFG_FILE" --ckpt_dir "$CKPT_DIR" --ckpt ${CKPT}