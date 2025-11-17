import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"
import torch.nn.functional
import numpy as np
from models.utils import *


class Encoder(nn.Module):
    def __init__(self, feat_channel=3, out_dim=1024):
        super().__init__()
        self.feat_channel = feat_channel
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, feat_channel, [64, 128], group_all=False, if_bn=False)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        l1_xyz, l1_points = self.sa_module_1(point_cloud, point_cloud)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        _, l3_points = self.sa_module_3(l2_xyz, l2_points)
        return l2_points, l3_points


class PrototypeGuidedTransformer(nn.Module):
    def __init__(self, class_num=16, dim_feat=1024, radius=1, probability=0.9, up_factor=1, i=0):
        super().__init__()
        self.i = i
        self.radius = radius
        self.label_prob = probability
        self.up_factor = up_factor

        # geometry branch
        self.xyz_mlp_1 = MLP_CONV(3, [64, 128])
        self.xyz_mlp_2 = MLP_CONV(128 * 2 + dim_feat + class_num, [256, 128])
        self.xyz_skip_transformer = SkipTransformer(128, 3, 64)
        self.xyz_mlp_ps = MLP_CONV(128, [64, 32])
        self.xyz_ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)
        self.xyz_up_sampler = nn.Upsample(scale_factor=up_factor)
        self.xyz_mlp_delta_feature = MLP_Res(256, 128, 128)
        self.xyz_mlp_delta = MLP_CONV(128, [64, 3])

        # semantic branch
        self.label_mlp_1 = MLP_CONV(class_num, [64, 128])
        self.label_mlp_2 = MLP_CONV(128 * 2 + class_num, [256, 128])
        self.label_skip_transformer = SkipTransformer(128, class_num, 64)
        self.label_mlp_ps = MLP_CONV(128, [64, 32])
        self.label_ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)
        self.label_up_sampler = nn.Upsample(scale_factor=up_factor)
        self.label_mlp_delta_feature = MLP_Res(256, 128, 128)
        self.label_mlp_delta = MLP_CONV(128, [64, class_num])

        self.prototype_mlp_delta = MLP_CONV(128, [64, class_num])

    def forward(self, feat_global, pcd_label, prototype, k_prev=None):
        _, _, n = pcd_label.shape
        xyz = pcd_label[:, :3, :].contiguous()
        label = pcd_label[:, 3:, :].contiguous()
        # geometry branch
        feat_xyz = self.xyz_mlp_1(xyz)
        feat_xyz_cat = torch.cat([feat_xyz,
                                  torch.max(feat_xyz, dim=2, keepdim=True)[0].repeat(1, 1, n),
                                  prototype,
                                  feat_global.repeat(1, 1, n)], dim=1)
        query_xyz = self.xyz_mlp_2(feat_xyz_cat)

        value_xyz = self.xyz_skip_transformer(xyz,
                                              k_prev if k_prev is not None else query_xyz,
                                              query_xyz)
        feat_value_xyz = self.xyz_mlp_ps(value_xyz)
        feat_value_xyz = self.xyz_ps(feat_value_xyz)
        value_xyz_up = self.xyz_up_sampler(value_xyz)

        k_curr = self.xyz_mlp_delta_feature(torch.cat([feat_value_xyz, value_xyz_up], dim=1))
        delta_xyz = torch.tanh(self.xyz_mlp_delta(torch.relu(k_curr))) / (self.radius ** self.i)
        xyz_up = self.xyz_up_sampler(xyz)
        xyz_fine = xyz_up + delta_xyz

        # semantic branch
        feat_label = self.label_mlp_1(label)
        feat_label_cat = torch.cat([feat_label,
                                    torch.max(feat_label, dim=2, keepdim=True)[0].repeat(1, 1, n),
                                    prototype
                                    ], dim=1)
        value_label = self.label_mlp_2(feat_label_cat)

        feat_value_label = self.label_mlp_ps(value_label)
        feat_value_label = self.label_ps(feat_value_label)
        label_up = self.label_up_sampler(label)
        feat_value_label = self.label_skip_transformer(label_up,
                                                       feat_value_label,
                                                       k_curr)

        k_curr_ = self.label_mlp_delta_feature(torch.cat([feat_value_label, feat_value_label], dim=1))
        delta_label = torch.tanh(self.label_mlp_delta(torch.relu(k_curr_))) / (self.label_prob ** self.i)
        label_fine = label_up + delta_label

        delta_proto = self.prototype_mlp_delta(torch.relu(k_curr_))
        prototype_ = self.label_up_sampler(prototype) + delta_proto

        return torch.cat([xyz_fine, label_fine], dim=1), prototype_, k_curr_


class Decoder(nn.Module):
    def __init__(self, class_num=16, dim_feat=1024, radius=1.2, up_factors=(1, 2), probability=0.9):
        super().__init__()
        self.cls = class_num
        self.up_factors = list(up_factors)
        self.uppers = nn.ModuleList([PrototypeGuidedTransformer(class_num,
                                                                dim_feat,
                                                                radius,
                                                                probability,
                                                                factor,
                                                                i) for i, factor in enumerate(self.up_factors)])

    def forward(self, feat, pcd_label, prototype):
        list = []
        k_prev = None
        for i, upper in enumerate(self.uppers):
            pcd_label, prototype, k_prev = upper(feat, pcd_label, prototype, k_prev)
            list.append(pcd_label.permute(0, 2, 1).contiguous())
        return list


class ProtoFormer(nn.Module):
    def __init__(self, class_num=16, dim_feat=1024, num_coarse=2048, radius=1.2, up_factors=(1, 2)):
        super().__init__()
        self.cls = class_num
        self.num_coarse = num_coarse
        self.encoder = Encoder(3, dim_feat)
        self.decoder = Decoder(class_num, dim_feat, radius, up_factors)
        self.label_mlp = nn.Sequential(
            nn.Conv1d(dim_feat + 3, 256, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, self.cls, 1)
        )
        self.prototype_mlp = nn.Sequential(
            nn.Conv1d(class_num, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, num_coarse, 1)
        )

    def forward(self, pcd):
        indices = torch.arange(self.cls, device=pcd.device)
        semantic_one_hot = torch.nn.functional.one_hot(indices,
                                                     num_classes=self.cls).float().unsqueeze(0).repeat(pcd.size(0), 1, 1)
        _, feat_g = self.encoder(pcd.permute(0, 2, 1).contiguous())
        coarse = fps_subsample(pcd, self.num_coarse)
        label = self.label_mlp(torch.cat([coarse.permute(0, 2, 1).contiguous(),
                                          feat_g.repeat(1, 1, coarse.shape[1])], 1))
        prototype = self.prototype_mlp(semantic_one_hot)
        pcd_label = torch.cat([coarse.transpose(1, 2), label], 1)
        return self.decoder(feat_g, pcd_label, prototype.transpose(1, 2).contiguous())


if __name__ == '__main__':
    x_part = torch.randn(4, 4096, 3).cuda()
    # model = ProtoFormer(16,1024,2048,1.2,[1,2]).cuda() # 3,752,278
    model = ProtoFormer(12, 1024, 1024, 1, [2, 2, 2]).cuda() # 4,509,597
    out = model(x_part)
    for i in range(len(out)):
        print(out[i].shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")