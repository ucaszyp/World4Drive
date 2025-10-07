import torch, random
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F
from torch.linalg import inv
import math

from custom_mmdet3d.models import builder
from custom_mmdet3d.models.builder import build_loss
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from .CrossAttention import CrossAttention

from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time, json

from projects.mmdet3d_plugin.W4D.dense_heads.utils import get_locations, get_locations_reso
from projects.mmdet3d_plugin.W4D.dense_heads.pos3d import Ego3DPositionEmbeddingMLP
# from thop import profile

@HEADS.register_module()
class SimpleWayDecoderHead(BaseModule):
    def __init__(self,
                num_proposals=6,
                #MHA
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.0,
                #pos embedding
                depth_step=0.8,
                depth_num=64,
                depth_start = 0,
                position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                stride=32,
                num_views=6,
                #others
                train_cfg=None,
                test_cfg=None,
                use_wm=True,
                num_spatial_token=240,
                num_tf_layers=3,
                num_traj_modal=1,   # cfg is 3, 3 for 3 cmd
                num_mode=6, # 6 for 6 mode
                **kwargs,
                ):
        """
        use to predict the waypoints
        """
        super().__init__(**kwargs)
        self.use_wm = use_wm

        # query feature
        self.num_views = num_views
        self.num_proposals = num_proposals
        self.prev_window_len=6
        self.prev_view_feat_window=None
        # self.waypoint_query_feat = nn.Parameter(torch.randn(1, self.num_proposals, hidden_channel))
        self.waypoint_query_feat = nn.Parameter(torch.randn(1, hidden_channel)) # [1,256]

        # deconv for mask
        self._deconv = nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=4, stride=4, padding=0)
        self._semantic_head = nn.Sequential(
            nn.Conv2d(
                hidden_channel,
                hidden_channel,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channel,
                17,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(384, 640),
                mode="bilinear",
                align_corners=False,
            ),
        )

        # temp attn
        temp_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        ) 
        
        self._temp_decoder = nn.ModuleList( [
            nn.TransformerDecoder(temp_decoder_layer, num_tf_layers) 
            for _ in range(self.num_views)])

        # wp_attn
        wp_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.wp_attn = nn.TransformerDecoder(wp_decoder_layer, num_tf_layers) # input: Bz, num_token, d_model

        # ego_self_attn
        ego_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.ego_attn = nn.TransformerEncoder(ego_encoder_layer, 1) # input: Bz, num_token, d_model 

        # world model
        self._wm_query_embedding = nn.ModuleList( [
            nn.Embedding(num_spatial_token, hidden_channel)
            for _ in range(self.num_views)])
        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._wm_decoder = nn.ModuleList( [
            nn.TransformerDecoder(wm_decoder_layer, num_tf_layers) 
            for _ in range(self.num_views)])

        self.wp_encoder = nn.Sequential(
            nn.Linear(2*6, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )
        
        # loss
        self.loss_plan_reg = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.loss_plan_cls = build_loss(dict(type='FocalLoss',  
                                             use_sigmoid=True,
                                             gamma=2.0,
                                             alpha=0.25,
                                             loss_weight=0))
        self.loss_wm_cls = build_loss(dict(type='FocalLoss',
                                            use_sigmoid=True,
                                            gamma=2.0,
                                            alpha=0.25,
                                            loss_weight=0.5))  
        self.loss_plan_rec = nn.MSELoss()

        # head
        self.num_traj_modal = num_traj_modal
        self.num_mode = num_mode  # 6
        self.waypoint_head = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel * 2, hidden_channel * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel * 2, hidden_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel // 2, self.num_proposals * 2) # 6*2
            )
        self.waypoint_cls_head = nn.Sequential(
                # nn.Flatten(start_dim=1),    #[1, 6, 256] -> [1, 6*256]
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel // 2, hidden_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel // 2, 1) # 1
            )
        
        self.wm_cls_head = nn.Sequential(
                # nn.Flatten(start_dim=1),    #[1, 6, 256] -> [1, 6*256]
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel // 2, hidden_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel // 2, 1) # 1
            )
        
        self.waypoint_mode_embedding = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),)
        
        ego_motion_mode_ref = np.load('data/kmeans/kmeans_plan_6.npy')
        self.waypoint_mode_ref = nn.Parameter(
            torch.tensor(ego_motion_mode_ref, dtype=torch.float32),
            requires_grad=False,
        )

        self.ego_pos_embedding = nn.Linear(2, hidden_channel)
        
        
        # position embedding
        ##img pos embed
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride

        self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
            2**2 * 3, num_pos_feats=hidden_channel, n_freqs=8
        )
        
        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, hidden_channel*4),
                nn.ReLU(),
                nn.Linear(hidden_channel*4, hidden_channel),
            )

        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # LID depth
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)


    def prepare_location(self, img_metas, img_feats):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = img_feats.shape[:2]
        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location
    
    def backproject_patch(self, K: torch.Tensor, lidar2cam: torch.Tensor, depth: torch.Tensor, patch_size=32, reso=2) -> torch.Tensor:
        """
        Backproject depth map to 3D points in camera coordinate.
        Args:
            K: camera intrinsic matrix (b 3 3)
            depth: depth map (b 1 h w)
            pixel_offset: offset to the pixel coordinate
        """
        # __import__("ipdb").set_trace()
        b = len(depth)
        c, h, w = depth[0].shape
        hp, wp = h // patch_size, w // patch_size
        sub_hp = sub_wp = reso
        depth = torch.stack(depth, dim=0).detach()
        device = depth.device
        patch_depth = torch.nn.functional.interpolate(depth, size=(hp * reso, wp * reso), mode="area").reshape(b, c, -1)
        coords = get_locations_reso(hp, wp, device, self.stride, reso)
        p_cam = (inv(K.float()) @ coords.float()) * patch_depth  # (b 3 3) @ (3 hw) -> (b 3 hw) * (b 1 hw) -> (b 3 hw)
        p_cam = torch.cat([p_cam, torch.ones_like(p_cam[:, 0:1, :])], dim=1)
        p_lidar = (inv(lidar2cam) @ p_cam)[:, 0:3, :]
        patch_p_lidar = p_lidar.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
        return patch_p_lidar
    
    def img_position_embeding(self, img_feats, img_metas):
        """
        from streampetr
        """
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views, 'num_views should be equal to self.num_views'
        BN = B * num_views
        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        # Depth
        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        lidar2img = torch.from_numpy(np.stack(img_metas[0]['lidar2img'])).to(img_feats.device).float()
        lidar2img = lidar2img[:num_views]
        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(num_views, 1, 1, 4, 4).repeat(B, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3]) #normalize
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d) #(B, num_views*H*W, 3*64)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding
    
    def gen_sineembed_for_position(self, pos_tensor, hidden_dim=256):
        """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
        """
        half_hidden_dim = hidden_dim // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
        x_embed = pos_tensor[..., 0] * scale
        y_embed = pos_tensor[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos
    
    def forward(self, img_feat, img_metas, ego_info=None, depth=None, img=None, is_test=False, is_vis=True):
        # init
        losses = {}
        Bz, num_views, num_channels, height, width = img_feat.shape

        upscale_img_feat=[self._deconv(i) for i in img_feat.permute(1,0,2,3,4)]
        semantic_pred=[self._semantic_head(upscale_i) for upscale_i in upscale_img_feat]

        init_waypoint_query_feat = self.waypoint_query_feat.clone().unsqueeze(1).repeat(Bz, 1, 1)    # [1,256] -> [Bz,1,256]

        # img pos emb
        lidar2img = torch.from_numpy(np.stack(img_metas[0]['lidar2img'])).to(img_feat.device).float().detach()
        lidar2cam = torch.from_numpy(np.stack(img_metas[0]['lidar2cam'])).to(img_feat.device).float().detach()
        cam2lidar = lidar2cam.inverse().detach()
        K = torch.matmul(lidar2img, cam2lidar)[:, 0:3, 0:3].detach()
        xyz = self.backproject_patch(
            K, lidar2cam, depth, patch_size=self.stride, reso=2
        )
        img_pos3d = self.position_embedding_3d(xyz)
        img_pos3d = img_pos3d.unsqueeze(0).reshape(Bz, num_views, height, width, num_channels)
        img_pos3d = img_pos3d.permute(0, 1, 4, 2, 3)
        img_feat_emb = img_feat + img_pos3d

        spatial_view_feat = img_feat_emb.reshape(Bz, num_views, num_channels, height*width).permute(0, 1, 3, 2)

        if self.prev_view_feat is None:
            self.prev_view_feat = torch.zeros_like(spatial_view_feat)
        
        final_view_feat = torch.zeros_like(spatial_view_feat)
        for i in range(self.num_views):
            final_view_feat[:, i] = self._temp_decoder[i](spatial_view_feat[:, i], self.prev_view_feat[:, i])
                        #bz, num_views, num_token, num_channel
        self.prev_view_feat = final_view_feat.detach()
        final_view_feat = final_view_feat.reshape(Bz, -1, num_channels)


        if self.num_traj_modal > 1:
            waypoint_mode_ref = self.waypoint_mode_ref[None].repeat(Bz, 1, 1, 1, 1)  # [1, 3, 6, 6, 2]
            bz, num_traj_modal, num_mode, traj_len, _ = waypoint_mode_ref.shape
           
            waypoint_mode_query = self.gen_sineembed_for_position(waypoint_mode_ref[..., -1, :])    #[B,3,6,6,2]->[B,3,6,256]
            
            waypoint_mode_query = self.waypoint_mode_embedding(waypoint_mode_query)  # [B,3,6,256]


            waypoint_mode_query = waypoint_mode_query.flatten(1,2)    # [B,3,6,256]->[B,18,256]
            updated_waypoint_query_feat = init_waypoint_query_feat + waypoint_mode_query    # [B,3*6,256]
            
            
            ego_pos = torch.zeros((Bz, 1, 2), device=updated_waypoint_query_feat.device)    
            ego_pos_emb = self.ego_pos_embedding(ego_pos)
            updated_waypoint_query_feat = updated_waypoint_query_feat + ego_pos_emb 
            updated_waypoint_query_feat = self.ego_attn(updated_waypoint_query_feat)    # [b, 3*6, 256]

            updated_waypoint_query_feat = self.wp_attn(updated_waypoint_query_feat, final_view_feat)   #[1, 3*6, 256] 只接受[bs, num_tokens, feat_dim]的输入

            updated_waypoint_query_feat = updated_waypoint_query_feat.reshape(updated_waypoint_query_feat.shape[0], self.num_traj_modal, self.num_mode, -1)
            
            cur_waypoint = self.waypoint_head(updated_waypoint_query_feat)  # [B,3,6,256] -> [B,3,6,6*2]

            # bz, num_traj_modal, num_mode, _ = cur_waypoint.shape
            cur_waypoint = cur_waypoint.reshape(bz, num_traj_modal, num_mode, traj_len, 2)   #[1, 3, 6, 6, 2]
            # 计算每个模态的置信度
            cur_waypoint_cls = self.waypoint_cls_head.to(updated_waypoint_query_feat.device)(updated_waypoint_query_feat)  # [1,3,6,1]
            cur_waypoint_cls = cur_waypoint_cls.reshape(bz, num_traj_modal, num_mode)  # [1, 3, 6]
            ego_cmd = img_metas[0]['ego_fut_cmd'].to(img_feat.device)[0, 0] # tensor([3])
            cur_waypoint = cur_waypoint[:, ego_cmd == 1].squeeze(1) # [1, 3, 6, 6, 2] -> [1, 1, 6, 6, 2] -> [1, 6, 6, 2] 
            cur_waypoint_cls = cur_waypoint_cls[:, ego_cmd == 1].squeeze(1)  # [1, 3, 6] -> [1, 6] 
            best_traj_idx = torch.argmax(cur_waypoint_cls, dim=1)  # tensor([1])
            best_traj = cur_waypoint[:, best_traj_idx].squeeze(1)  # [1, 6, 2] 
           
        # world model prediction

        assert cur_waypoint.shape[1] == num_mode
        pred_img_feat_list = []
        for i in range(num_mode):
            # pred_img_feat = self.wm_prediction(spatial_view_feat, cur_waypoint[:, i])
            pred_img_feat = self.wm_prediction(spatial_view_feat, cur_waypoint.cumsum(-2)[:, i])
            # pred_img_feat = self.wm_prediction(spatial_view_feat, waypoint_mode_ref[:, ego_cmd == 1].squeeze(1)[:, i])
            pred_img_feat_list.append(pred_img_feat)
        pred_img_feat = torch.stack(pred_img_feat_list, dim=1)  # [1, 6, 6, 240, 256] 第一个6表示6个模态，第二个6是6个视角
        pred_img_cls_feat = pred_img_feat.mean(dim=[3, 2])  # [1, 6, 6, 240, 256] -> [1, 6, 256]
        pred_img_cls = self.wm_cls_head(pred_img_cls_feat)  # [1, 6, 256] -> [1, 6, 1]


        return cur_waypoint, cur_waypoint_cls, spatial_view_feat, pred_img_feat, pred_img_cls, semantic_pred
    


    def upscale(self, spatial_view_feat,shape_):
        # # 反卷积操作
        Bz, num_views, num_channels, height, width=shape_
        spatial_view_feat=spatial_view_feat.reshape(Bz,num_views,height,width,num_channels)
        spatial_view_feat = spatial_view_feat.permute(0, 1, 4, 2, 3)  # 变为 [1, 6, 256, H, W]
        spatial_view_feat = spatial_view_feat.reshape(Bz * num_views, num_channels, height, width)  # 变为 [6, 256, H, W]
        upscaled_feat = self._deconv(spatial_view_feat)  # 输出 [6, 256, 4H, 4W]

        # 调整回原始维度
        upscaled_feat = upscaled_feat.reshape(Bz, num_views, num_channels, 4*height, 4*width)  # 变回 [1, 6, 256, 4H, 4W]
        upscaled_feat = upscaled_feat.permute(0, 1, 3, 4, 2)  # 变回 [1, 6, 4H, 4W, 256]
        spatial_view_feat=upscaled_feat.reshape(Bz,num_views,16*height*width,num_channels)

        return spatial_view_feat

    def loss_semantic(self,semantic_pred,mask):
        loss=[]
        for i in range(len(semantic_pred)):
            loss.append(F.cross_entropy(semantic_pred[i],mask[i].long()))
        loss=sum(loss)/len(loss)
        return loss

    def loss_reconstruction(self, 
            reconstructed_view_query_feat,
            observed_view_query_feat,
            mask=None,
            ):

        loss_rec = self.loss_plan_rec(reconstructed_view_query_feat, observed_view_query_feat)

        return loss_rec
    
    def loss_kl(self, pred_img_feat, cur_img_feat, epsilon=1e-10):
        """
        计算 KL 散度 Loss
        参数:
        pred    - 预测张量，形状为 [1, 6, 240, 256]
        target  - 真实张量，形状为 [1, 6, 240, 256]
        epsilon - 防止 log(0) 数值问题的小常数（默认 1e-10）
        返回:
        一个标量，表示所有 token 和视角上的平均 KL 散度 Loss
        """
        # 将最后一维（256）归一化为概率分布
        pred_img_feat_prob = F.softmax(pred_img_feat, dim=-1)
        cur_img_feat_prob = F.softmax(cur_img_feat, dim=-1)
        
        # 加入 epsilon 以避免 log(0)
        pred_img_feat_prob = pred_img_feat_prob + epsilon
        cur_img_feat_prob = cur_img_feat_prob + epsilon
        
        # 根据 KL 散度公式计算每个 token 的散度：P * log(P / Q)
        kl_div = pred_img_feat_prob * torch.log(pred_img_feat_prob / cur_img_feat_prob)
        
        # 沿最后一维（256）求和，得到每个 token 的 KL 值，
        # 然后对 batch、视角和 token 求平均
        kl_loss = kl_div.sum(dim=-1).mean()
        return kl_loss
    
    def loss_cosine(self, pred, target):
        """
        计算余弦相似度损失
        参数:
        pred   - 预测张量，形状为 [1,6,240,256]
        target - 真实张量，形状为 [1,6,240,256]
        返回:
        一个标量，表示所有视角和 token 上的平均余弦相似度损失
        """
        # 计算余弦相似度，沿最后一维（256）计算，结果形状为 [1,6,240]
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        
        # 定义余弦相似度损失，常用形式是 1 - cosine similarity
        cosine_loss = 1 - cos_sim
        
        # 对所有视角和 token 求均值，得到最终损失
        loss = cosine_loss.mean()
        return loss
    
    def loss_wm_diversity(self, pred_img_feat, cosine_thre=0):
        B, num_mode, num_views, num_tokens, feat_dim = pred_img_feat.shape
        features_norm = F.normalize(pred_img_feat, p=2, dim=-1)
        diversity_loss = 0.0
        count = 0
        
        for b in range(B):
            for i in range(num_mode):
                for j in range(i + 1, num_mode):
                    # 计算归一化后的平均余弦相似度
                    cosine_sim = torch.sum(features_norm[b, i] * features_norm[b, j]) / (num_views * num_tokens)
                    # 仅惩罚正值相似度
                    diversity_loss += torch.clamp(cosine_sim - cosine_thre, min=0)
                    count += 1
                    
        if count == 0:
            return torch.tensor(0.0, device=pred_img_feat.device)
        diversity_loss /= count
        return diversity_loss
    
    def loss_traj_diversity(self, 
                            pred_trajs,
                            edl_weight=0.0,
                            kld_weight=0.5,
                            bce_weight=0.1):
        """
        综合轨迹终点多样性损失，包含：
        1. 终点欧几里得距离损失 (Endpoint Diversity Loss)
        2. 终点 KL 散度损失 (Endpoint KL Divergence Loss)
        3. 终点 BCE 损失 (Endpoint BCE Loss)
        """
        pred_trajs = pred_trajs.cumsum(-2)
        loss_endpoint_div = endpoint_diversity_loss(pred_trajs)
        loss_endpoint_kl = endpoint_kl_divergence(pred_trajs)
        loss_endpoint_bce = endpoint_bce_loss(pred_trajs)

        # 组合损失
        total_loss = edl_weight * loss_endpoint_div + kld_weight * loss_endpoint_kl + bce_weight * loss_endpoint_bce

        return total_loss
    
    def wm_prediction(self, img_feat, cur_waypoint):
        batch_size = img_feat.shape[0]
        cur_waypoint = cur_waypoint.reshape(batch_size, 1, -1)
        wp_token = self.wp_encoder(cur_waypoint) # output: bz, 1, hidden_channel
        wm_next_latent = torch.zeros_like(img_feat)
        for i in range(self.num_views):
            input_tokens = torch.cat([img_feat[:, i], wp_token], dim=1)
            wm_query = self._wm_query_embedding[i].weight[None, ...].repeat(batch_size, 1, 1)
            wm_next_latent[:, i] = self._wm_decoder[i](wm_query, input_tokens)
        return wm_next_latent
    
    def loss_3d(self, 
            preds_ego_future_traj,  #[1,6,6,2]
            preds_ego_future_traj_cls,  #[1,6]
            best_wm_idx,  #[1]
            gt_ego_future_traj, #[1,6,2]
            gt_ego_future_traj_mask,    #[1,6,1]
            ego_info=None,
            ):
        ego_future_label = best_wm_idx
        best_ego_future_traj = preds_ego_future_traj[:, ego_future_label].reshape(1, 6, 2)  # [1, 6, 2]
        loss_waypoint = self.loss_plan_reg(best_ego_future_traj, gt_ego_future_traj, gt_ego_future_traj_mask)
        loss_waypoint_cls = self.loss_plan_cls(preds_ego_future_traj_cls, ego_future_label)
        return loss_waypoint, loss_waypoint_cls
    
    def get_waypoint_label(self, 
            preds_ego_future_traj,
            gt_ego_future_traj,
            gt_ego_future_traj_mask=None,
            ):
        cum_traj_preds = preds_ego_future_traj.cumsum(dim=-2)    
        cum_traj_targets = gt_ego_future_traj.cumsum(dim=-2)
        if gt_ego_future_traj_mask is None:
            gt_ego_future_traj_mask = torch.ones_like(gt_ego_future_traj[..., :1])
        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)  # [1,6,6]
        dist = dist * gt_ego_future_traj_mask[:, None, :]   # [1,1,6,6]

        dist = dist[..., :, -1] # 

        traj_labels = torch.argmin(dist, dim=-1)[0]    #[1]
        return traj_labels
    
    def select_optimal_modality(self, loss_rec_tensor, loss_kl_tensor, loss_cos_tensor, prev_pred_trajs, gt_trajs, wm_loss_weight=1.0, weight_tm=0.01):  #0.2 0.1
        """
        综合重建损失和轨迹匹配损失选择最优模态。
        
        参数：
        prev_pred_img_feat: 前一帧预测得到的各模态特征, [B, num_modalities, feat_dim]
        cur_img_feat: 当前帧图像特征, [B, feat_dim]
        preds_ego_future_traj: 当前帧预测的未来轨迹, [B, num_modalities, num_timesteps, 2]
        gt_trajs: 当前帧GT轨迹, [B, num_timesteps, 2]
        wm_loss_weight: 重建损失的权重（例如0.2）
        weight_tm: 轨迹匹配损失的缩放因子（需要根据实际量级调节）
        
        返回：
        best_modality_idx: [B]，每个样本选择的模态索引
        rec_loss_tensor: [B, num_modalities] 重建损失（未经加权）
        tm_loss_tensor:  [B, num_modalities] 轨迹匹配损失（未经加权）
        total_loss:     [B, num_modalities] 综合损失
        """
        prev_pred_trajs = prev_pred_trajs.cumsum(-2)
        gt_trajs = gt_trajs.cumsum(-2)
        B, num_modalities, _, _ = prev_pred_trajs.shape
        # 1. 计算重建损失：每个模态比较前一帧预测特征与当前帧图像特征。
        loss_rec = loss_rec_tensor + loss_kl_tensor + loss_cos_tensor
        
        # 2. 计算轨迹匹配损失：以最终位置误差（Final Displacement Error, FDE）为例
        #    这里preds_ego_future_traj: [B, num_modalities, num_timesteps, 2]
        #         gt_trajs: [B, num_timesteps, 2]
        tm_loss_list = []
        for i in range(num_modalities):
            pred_traj_i = prev_pred_trajs[:, i, :, :]  # [B, num_timesteps, 2]
            # 计算终点欧几里得距离
            fde = torch.norm(pred_traj_i[:, -1, :] - gt_trajs[:, -1, :], dim=-1)  # [B]
            tm_loss_list.append(fde)
        tm_loss_tensor = torch.cat(tm_loss_list, dim=0)  # [num_modalities]
        
        # 3. 综合损失：注意量级不一致，所以用权重进行平衡
        total_loss = loss_rec * wm_loss_weight + tm_loss_tensor * weight_tm  # [num_modalities]
        
        # 4. 对于每个样本，选择综合损失最小的模态
        best_modality_idx = total_loss.argmin(dim=-1).reshape(1)  # [B]
        
        return best_modality_idx
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def endpoint_diversity_loss(pred_trajs):
    """
    计算所有轨迹的终点之间的欧几里得距离，使得不同模态的终点位置尽可能分散。
    
    pred_trajs: 预测轨迹 [B, num_modalities, num_timesteps, 2]
    """
    B, num_modalities, num_timesteps, _ = pred_trajs.shape
    # 获取轨迹的终点坐标
    traj_endpoints = pred_trajs[:, :, -1, :]  # [B, num_modalities, 2]

    diversity_loss = 0.0
    count = 0

    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            endpoint_dist = torch.norm(traj_endpoints[:, i] - traj_endpoints[:, j], dim=-1)  # [B]
            diversity_loss += 1.0 / (endpoint_dist + 1e-5)  # 反向约束，距离越小，损失越大
            count += 1

    diversity_loss = diversity_loss / count
    return diversity_loss

def endpoint_kl_divergence(pred_trajs):
    """
    计算不同轨迹的终点分布，利用 KL 散度约束它们的分布尽可能均匀。
    
    pred_trajs: 预测轨迹 [B, num_modalities, num_timesteps, 2]
    """
    B, num_modalities, num_timesteps, _ = pred_trajs.shape
    traj_endpoints = pred_trajs[:, :, -1, :]  # [B, num_modalities, 2]

    # 计算终点的均值和方差
    mean = traj_endpoints.mean(dim=1, keepdim=True)  # [B, 1, 2]
    std = traj_endpoints.std(dim=1, keepdim=True) + 1e-5  # [B, 1, 2]

    # 计算每个轨迹终点的归一化概率分布
    endpoint_dist = (traj_endpoints - mean) / std  # [B, num_modalities, 2]
    endpoint_prob = F.softmax(-endpoint_dist.norm(dim=-1), dim=1)  # [B, num_modalities]

    # 计算 KL 散度损失
    uniform_dist = torch.ones_like(endpoint_prob) / num_modalities
    kl_loss = F.kl_div(endpoint_prob.log(), uniform_dist, reduction='batchmean')

    return kl_loss

def endpoint_bce_loss(pred_trajs):
    """
    计算轨迹终点的二分类交叉熵损失，使得所有轨迹的终点在预测分布上尽可能均匀。
    
    pred_trajs: 预测轨迹 [B, num_modalities, num_timesteps, 2]
    """
    B, num_modalities, num_timesteps, _ = pred_trajs.shape
    traj_endpoints = pred_trajs[:, :, -1, :].contiguous()  # [B, num_modalities, 2] 这里一定要使用.contiguous()，否则下面cdist会报错

    # 计算轨迹终点的 pairwise 距离
    pairwise_distances = torch.cdist(traj_endpoints, traj_endpoints, p=2)  # [B, num_modalities, num_modalities]

    # 设定一个最小终点距离阈值 (0.5m)
    min_dist = 0.5

    # 计算终点距离的 Binary Cross-Entropy (BCE) 损失
    target = (pairwise_distances > min_dist).float()  # 期望终点距离尽可能大
    bce_loss = F.binary_cross_entropy_with_logits(pairwise_distances, target)

    return bce_loss
        
