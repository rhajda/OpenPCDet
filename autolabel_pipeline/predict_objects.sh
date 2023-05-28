
#!/usr/bin/env bash

'''
Triggers predict_objects.py
'''


# Assign arguments to the variables
CFG_FILE="/home/tools/cfgs/autolabel_models/pointrcnn.yaml"
CKPT_DIR="/home/output/autolabel_models/pointrcnn/default/ckpt/"
CKPT='checkpoint_epoch_74.pth'

# Run:
python predict_objects.py --cfg_file "$CFG_FILE" --ckpt_dir "$CKPT_DIR" --ckpt ${CKPT}
