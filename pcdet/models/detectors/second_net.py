from .detector3d_template import Detector3DTemplate
import torch


class SECONDNet(Detector3DTemplate):
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
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
