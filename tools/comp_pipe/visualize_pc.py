import os
import glob
import numpy as np
from pathlib import Path

try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from tools.visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
    
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from tools.comp_pipe.path_handle import paths

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            points = np.asarray(open3d.io.read_point_cloud(str(self.sample_file_list[index]), format="pcd").points)[:, :3]
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
def visualize_pointcloud(points: np.array, ref_boxes: np.array, ref_scores: np.array, 
                         ref_labels: np.array, just_image=True) -> None:
    """
    Visualize pointcloud using open3d or mayavi.
    """
    V.draw_scenes(points, ref_boxes, ref_scores, ref_labels,just_image=just_image)
    if not OPEN3D_FLAG:
        mlab.show(stop=True)
        
def main():
        
    data_path = paths.indy_exp_real
    
    # Avoids always using '--ext' when is is clear from the given path
    ext='.bin'
    if Path(data_path).is_file:
        ext = str(Path(data_path).suffix) 

    # set data config
    os.chdir(paths.tools)
    cfg_from_yaml_file(paths.cfg_indy_pointrcnn, cfg)
    os.chdir(paths.root)
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext
        )
        
    # Display raw data
    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    
    visualize_pointcloud(
        points=data_dict['points'][:, 1:], ref_boxes=None,
        ref_scores=None, ref_labels=None
    )
    
    # Display preprocessed/augmented data
    
    # Display ground_truths
    
    # Display predictions
    
if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Running...")
    main()
    print("Time: {:.2f}s".format(time.time() - t0))
    