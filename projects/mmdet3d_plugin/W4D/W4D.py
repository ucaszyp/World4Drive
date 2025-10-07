import time
import copy
import numpy as np
import torch, cv2, os, random
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from custom_mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from custom_mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import VAD
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric
from custom_mmdet3d.models import builder

import matplotlib.pyplot as plt

@DETECTORS.register_module()
class W4D(VAD):
    def __init__(self,
                use_video=False,
                use_swin=False,
                only_front_view=False,
                use_multi_view=True,
                swin_input_channel=768,
                hidden_channel=256,
                use_semantic=False,
                semantic_img_backbone=None,
                flow_only=False,
                all_zero=False,
                semantic_only=False,
                use_2d_waypoint=False,
                wm_loss_weight=0.2,
                **kwargs,
                 ):
        super().__init__( **kwargs)
        self.depth_eps = 2.75
        self.ref_pts_cam_list = []
        self.use_video = use_video
        self.use_swin = use_swin
        self.use_semantic = use_semantic

        self.flow_only = flow_only
        self.semantic_only = semantic_only
        self.all_zero = all_zero

        self.only_front_view = only_front_view
        self.use_2d_waypoint = use_2d_waypoint

        if semantic_img_backbone is not None:
            self.semantic_img_backbone = builder.build_backbone(semantic_img_backbone)

        if (not self.with_img_neck) and self.use_swin:
            self.swin_img_mlp = nn.Linear(swin_input_channel, hidden_channel)
        
        self.metrics_history = []
        self.call_count = 0
        self.wm_loss_weight = wm_loss_weight
        self.semantic_loss_weight=0.05
        

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # My modification
            if self.use_video:
                img = img.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                img_feats = self.img_backbone(img)
                img_feats = img_feats.mean(dim=2, keepdim=True)
            else:
                if self.use_swin:
                    img = img.unsqueeze(2)
                img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if not isinstance(img_feats, tuple):
            img_feats = [img_feats.squeeze(2)]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        elif self.use_swin: #swin without fpn
            img_feats = [self.swin_img_mlp(img_feats[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def obtain_history_feat(self, imgs_queue, img_metas_list, depth=None, is_test=False, is_vis=False):
        """Obtain history BEV features iteratively.
        """
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        losses = {}
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list][0]
            # pred_ego_fut_trajs, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas)
            pred_ego_fut_trajs, pred_ego_fut_trajs_cls, cur_img_feat, pred_img_feat, pred_img_cls, _ = self.pts_bbox_head(img_feats, img_metas, depth=depth, img=imgs_queue, is_vis=is_vis)

            # compute loss
            if not is_test:
                # loss waypoint
                gt_ego_fut_trajs = img_metas[0]['ego_fut_trajs'].to(img_feats.device)
                gt_ego_fut_masks = img_metas[0]['ego_fut_masks'].squeeze(0).unsqueeze(-1).to(img_feats.device)
                return losses, pred_img_feat, pred_img_cls, pred_ego_fut_trajs, pred_ego_fut_trajs_cls, gt_ego_fut_trajs, gt_ego_fut_masks
        return losses

    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      semantic_img=None,
                      flow_img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      mask=None,
                      depth=None,
                      ):
        """
        agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        """
        if self.only_front_view:
            img = img[:, :, 0:1, ...]

        len_queue = img.size(1)
        num_view = img.size(2)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        
        prev_depth = depth[0: num_view * (len_queue - 1)]
        depth = depth[num_view * (len_queue - 1):]


        self.pts_bbox_head.prev_view_feat = None
        if len_queue > 1:
            prev_frame_losses, pred_img_feat, pred_img_cls, prev_pred_ego_fut_trajs, prev_pred_ego_fut_trajs_cls, prev_gt_ego_fut_trajs, prev_gt_ego_fut_masks = self.obtain_history_feat(prev_img, prev_img_metas, prev_depth)  
        else:
            prev_frame_losses = {}

        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]            
        losses = self.forward_pts_train(cur_img_feats, 
                                        cur_img_metas,
                                        pred_img_feat=pred_img_feat,
                                        pred_img_cls=pred_img_cls,
                                        prev_pred_trajs=prev_pred_ego_fut_trajs,
                                        prev_pred_trajs_cls=prev_pred_ego_fut_trajs_cls,
                                        prev_gt_trajs=prev_gt_ego_fut_trajs,
                                        prev_gt_trajs_masks=prev_gt_ego_fut_masks,
                                        ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                        ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                        ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,mask=mask, depth=depth,
                                    )
        
        return losses

    def forward_pts_train(self,
                          img_feats,
                          img_metas,
                          pred_img_feat=None,
                          pred_img_cls=None,
                          prev_pred_trajs=None,
                          prev_pred_trajs_cls=None,
                          prev_gt_trajs=None,
                          prev_gt_trajs_masks=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                          mask=None,
                          depth=None,
                        ):
        """Forward function
        Args:
            ego_fut_cmd: [turn_left, turn_right, go_straight]
            ego_lcf_feat: (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度

        """
        #get the ego info   
        losses = {}
        B = ego_his_trajs.size(0)
        ego_his_trajs = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs, ego_lcf_feat, ego_fut_cmd], dim=1)
        
        prev_pred_img_feat = pred_img_feat
        prev_pred_img_cls = pred_img_cls
        prev_pred_trajs = prev_pred_trajs
        prev_pred_trajs_cls = prev_pred_trajs_cls
        prev_gt_trajs = prev_gt_trajs
        prev_gt_trajs_masks = prev_gt_trajs_masks

        preds_ego_future_traj, preds_ego_future_traj_cls, cur_img_feat, pred_img_feat, pred_img_cls, semantic_pred = self.pts_bbox_head(img_feats, img_metas, ego_info, depth)
        
        # world model loss
        loss_rec_list = []
        for i in range(preds_ego_future_traj.shape[1]):
            loss_rec = self.pts_bbox_head.loss_reconstruction(
                                    prev_pred_img_feat[:, i], 
                                    cur_img_feat.detach(),
                                    )
            loss_rec_list.append(loss_rec)
        # restruction loss
        loss_rec_tensor = torch.stack(loss_rec_list, dim=0)


        # wm kl loss
        loss_kl_list = []
        for i in range(preds_ego_future_traj.shape[1]):
            loss_kl = self.pts_bbox_head.loss_kl(
                                    prev_pred_img_feat[:, i], 
                                    cur_img_feat.detach(),
                                    )
            loss_kl_list.append(loss_kl)
        loss_kl_tensor = torch.stack(loss_kl_list, dim=0)

        
        # wm cosine loss
        loss_cos_list = []
        for i in range(preds_ego_future_traj.shape[1]):
            loss_cos = self.pts_bbox_head.loss_cosine(
                                    prev_pred_img_feat[:, i], 
                                    cur_img_feat.detach(),
                                    )
            loss_cos_list.append(loss_cos)
        loss_cos_tensor = torch.stack(loss_cos_list, dim=0)

        
        ## wm diversity loss
        cosine_thre = 0.7   
        prev_loss_wm_diversity_0 = self.pts_bbox_head.loss_wm_diversity(prev_pred_img_feat, cosine_thre=cosine_thre)
        losses['prev_loss_wm_diversity_0'] = prev_loss_wm_diversity_0
        
        cur_loss_wm_diversity = self.pts_bbox_head.loss_wm_diversity(pred_img_feat, cosine_thre=cosine_thre) 
        losses['loss_wm_diversity'] = cur_loss_wm_diversity
        

        
        # select the best wm and traj idx
        prev_best_idx = self.pts_bbox_head.select_optimal_modality(loss_rec_tensor, loss_kl_tensor, loss_cos_tensor, prev_pred_trajs, prev_gt_trajs, weight_tm=0.0)

   
        losses['loss_rec'] = (loss_kl_list[prev_best_idx] * self.wm_loss_weight)
        losses['loss_kl'] = (loss_kl_list[prev_best_idx] * self.wm_loss_weight)
        losses['loss_cos'] = (loss_cos_list[prev_best_idx] * self.wm_loss_weight)
        

        loss_wm_cls = self.pts_bbox_head.loss_wm_cls(prev_pred_img_cls.squeeze(-1), prev_best_idx)
        losses['loss_wm_cls'] = loss_wm_cls

        # 优化特定模态

        # waypoint loss
        prev_loss_waypoint, prev_loss_waypoint_cls = self.pts_bbox_head.loss_3d(prev_pred_trajs,
                                                           prev_pred_trajs_cls,
                                                           prev_best_idx,
                                                           prev_gt_trajs,
                                                           prev_gt_trajs_masks,
                                                           )
        
        # semantic loss
        loss_semantic=self.pts_bbox_head.loss_semantic(semantic_pred,mask)
        losses['loss_sematic'] = loss_semantic*self.semantic_loss_weight
        losses.update({
                f'prev_frame_loss_waypoint_0': prev_loss_waypoint,
                f'prev_frame_loss_waypoint_cls_0': prev_loss_waypoint_cls,
            })

        
        cur_pred_best_idx = prev_best_idx

        loss_waypoint, loss_waypoint_cls = self.pts_bbox_head.loss_3d(preds_ego_future_traj,    # [1, 6, 6, 2]
                                            preds_ego_future_traj_cls,   # [1, 6]
                                            cur_pred_best_idx,  # [1]
                                            ego_fut_trajs.squeeze(1),   # [1,1,6,2] -> [1, 6, 2]
                                            ego_fut_masks.squeeze(0).squeeze(0).unsqueeze(-1),  # [1,1,1,6] -> [1, 6, 1]
                                            )
        losses.update({
            'loss_waypoint': loss_waypoint,
            'loss_waypoint_cls': loss_waypoint_cls
        })
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        depth=None,
        **kwargs
    ):

        bbox_results = self.simple_test(
            img_metas=img_metas,
            img=img,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            depth=depth,
            **kwargs
        )

        return bbox_results
    
    def simple_test(
        self,
        img_metas,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        depth=None,
        **kwargs,
    ):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        self.pts_bbox_head.prev_view_feat = None
        
        num_view = img.size(2)
        
        prev_depth = depth[0]
        depth = depth[-1]


        if len_queue > 1:
            prev_frame_losses, pred_img_feat, pred_img_cls, prev_pred_ego_fut_trajs, prev_pred_ego_fut_trajs_cls, prev_gt_ego_fut_trajs, prev_gt_ego_fut_masks = self.obtain_history_feat(prev_img, prev_img_metas, prev_depth)


        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]  

        bbox_list = [dict() for i in range(len(img_metas))]
        metric_dict, plan_annos = self.simple_test_pts(
            cur_img_feats,
            cur_img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            fut_valid_flag=fut_valid_flag,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            depth=depth,
            pred_img_feat=pred_img_feat,
            cur_img=cur_img,
        )
        
        sample_token = cur_img_metas[0]['sample_idx']
        for result_dict in bbox_list:
            result_dict[sample_token] = {'metric_results':metric_dict,
                                         'plan_results': plan_annos}

        return bbox_list
    
    def simple_test_pts(
        self,
        img_feats,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        depth=None,
        pred_img_feat=None,
        cur_img=None,
    ):
        """Test function"""
        B = ego_his_trajs.size(0)
        ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)
        prev_pred_img_feat = pred_img_feat

        preds_ego_future_traj, preds_ego_future_traj_cls, cur_img_feat, _, pred_img_cls, _ = self.pts_bbox_head(
                                        img_feats, 
                                        img_metas,
                                        depth=depth, 
                                        img=cur_img,
                                    )
        
        # world model loss
        loss_rec_list = []
        for i in range(preds_ego_future_traj.shape[1]):
            loss_rec = self.pts_bbox_head.loss_reconstruction(
                                    prev_pred_img_feat[:, i], 
                                    cur_img_feat.detach(),
                                    )
            loss_rec_list.append(loss_rec)
        loss_rec_tensor = torch.stack(loss_rec_list, dim=0)


        # wm kl loss
        loss_kl_list = []
        for i in range(preds_ego_future_traj.shape[1]):
            loss_kl = self.pts_bbox_head.loss_kl(
                                    prev_pred_img_feat[:, i], 
                                    cur_img_feat.detach(),
                                    )
            loss_kl_list.append(loss_kl)
        loss_kl_tensor = torch.stack(loss_kl_list, dim=0)

        
        # wm cosine loss
        loss_cos_list = []
        for i in range(preds_ego_future_traj.shape[1]):
            loss_cos = self.pts_bbox_head.loss_cosine(
                                    prev_pred_img_feat[:, i], 
                                    cur_img_feat.detach(),
                                    )
            loss_cos_list.append(loss_cos)
        loss_cos_tensor = torch.stack(loss_cos_list, dim=0)

        all_loss = loss_rec_tensor + loss_kl_tensor + loss_cos_tensor
        best_prev_traj_idx = all_loss.argmin(dim=-1).reshape(1)  
        cur_pred_best_idx = best_prev_traj_idx

        # save planning info
        plan_annos = {
            'pred_ego_fut_trajs': preds_ego_future_traj,
            'gt_ego_fut_trajs': ego_fut_trajs[0,0][None],
            'ego_fut_cmd': ego_fut_cmd_,
            'pred_wm_cls': pred_img_cls,
            'all_loss': all_loss,
            'best_prev_traj_idx': best_prev_traj_idx,
            
        }
        
        preds_ego_future_traj = preds_ego_future_traj[:, cur_pred_best_idx].squeeze(1)
        
        

        with torch.no_grad():
            # pre-process
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = preds_ego_future_traj[0]
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]



            ego_fut_preds = ego_fut_preds.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_preds[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )

            #mid print
            # update metrics
            self.metrics_history.append(metric_dict_planner_stp3)
            self.call_count += 1



        return metric_dict_planner_stp3, plan_annos
    



