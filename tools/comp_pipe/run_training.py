"""
Helper class to comfortly run a training wiht openpcdet. 
"""
import sys
import os
from pcdet.datasets.indy.kitti_object_eval_python.evaluate import evaluate
from tools.comp_pipe.analyzable_dataset  import paths
import pcdet.datasets.indy.kitti_object_eval_python.kitti_common as kitti
from pcdet.datasets.indy.kitti_object_eval_python.eval import get_official_eval_result, get_coco_eval_result


evaluate(label_path=paths.indy_no_noise / 'gt_database',
         result_path=paths.pipe_results,
    label_split_file=paths.indy_no_noise / 'ImageSets/val.txt',
    current_class=0, coco=False)

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

det_path = paths.pipe_results
dt_annos = kitti.get_label_annos(det_path)
gt_path = paths.indy_no_noise / 'gt_database'
gt_split_file = paths.indy_no_noise / 'ImageSets/val.txt' # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer