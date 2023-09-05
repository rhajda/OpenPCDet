from .detector3d_template import Detector3DTemplate
import torch
import numpy as np


class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, tb_log=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.tb_log = tb_log
        self.idx = 0
        for module in self.module_list:
            if hasattr(module, "eval_mode"):
                module.eval_mode = self.eval_mode
            if hasattr(module, "test"):
                module.test = self.test

    def forward(self, batch_dict):
        for idx, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            if idx == 0:
                feat = batch_dict["point_features"].cpu().detach().numpy()
                feat = np.amax(np.stack(np.split(feat, batch_dict["batch_size"])),1)

        for key in batch_dict.keys():
            if isinstance(batch_dict[key], torch.Tensor) and self.tb_log is not None:
                self.tb_log.add_scalar(f'train/{key}', (batch_dict[key].cpu().detach().numpy().sum()), self.idx)
        self.idx += 1

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, feat

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
