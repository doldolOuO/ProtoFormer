from functools import partial
from torch import nn
from cuda.ChamferDistance import L1_ChamferDistance_w_idx, L2_ChamferDistance_w_idx
from pointnet2_ops.pointnet2_utils import gather_operation, furthest_point_sample
from loss.focal_loss import FocalLoss


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points)
    )
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def seg_loss(pred, gt, idx, weight):
    gt_label = gt[:, :, 3].reshape(-1, gt.shape[1]).gather(1, idx.long())
    pred_label = pred[:, :, 3:]
    ls_seg = FocalLoss(alpha=weight)(pred_label, gt_label)
    return ls_seg, pred_label, gt_label


class Loss_train(nn.Module):
    def __init__(self, seg_weight, sqrt=True):
        super(Loss_train, self).__init__()
        self.sqrt = sqrt
        self.cd_ls = L1_ChamferDistance_w_idx() if sqrt else L2_ChamferDistance_w_idx()
        self.seg_ls = partial(seg_loss, weight=seg_weight)

    def forward(self, pcds_pred, gt):
        """loss function
        Args
            pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
        """
        cd_ls, seg_ls = [], []
        cd_i, seg_i, pred_label, gt_label = None, None, None, None
        for i, p_i in enumerate(pcds_pred):
            gt_i = (
                gt
                if i == 0 or i == len(pcds_pred) - 1
                else fps_subsample(gt, p_i.shape[1])
            )
            cd_i, idx_i = self.cd_ls(
                p_i[:, :, :3].contiguous(), gt_i[:, :, :3].contiguous()
            )

            seg_i, pred_label, gt_label = self.seg_ls(p_i, gt_i, idx_i)
            if i > 0:
                cd_ls.append(cd_i)
                seg_ls.append(seg_i)
        if self.sqrt:
            loss_cmp, loss_seg = sum(cd_ls) * 1e3, sum(seg_ls) * 1e1
        else:
            loss_cmp, loss_seg = sum(cd_ls) * 1e4, sum(seg_ls) * 1e1
        loss_all = loss_cmp + loss_seg

        return {
            "sum_loss": loss_all,
            "last_cd": cd_i,
            "last_seg": seg_i,
            "pred_seg": pred_label,
            "gt_seg": gt_label,
        }