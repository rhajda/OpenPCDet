
#!/bin/bash



CFG_FILE="/home/tools/cfgs/autolabel_models/pointpillar.yaml"
CKPT="/home/autolabel_data/autolabel_retrain/models/pointpillar/checkpoint_epoch_75.pth"


# Change directory to /home/tools
cd /home/tools

# Run the Python script with the provided arguments
python test.py --cfg_file "$CFG_FILE" --ckpt "$CKPT"
