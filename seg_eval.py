import torch
import numpy as np
from cuda.ChamferDistance import ChamferDistanceFunctionWithIdxNoGrad

_dist_with_idx = None  # lazy init

def fast_hist(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (labels >= 0) & (labels < num_classes)
    hist = np.bincount(
        num_classes * labels[mask].astype(int) + preds[mask], minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def eval_points_segmentation(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    pred_seg: torch.Tensor,
    gt_seg: torch.Tensor,
    num_classes: int,
    ignore_index: int,
):
    global _dist_with_idx
    if _dist_with_idx is None:
        _dist_with_idx = ChamferDistanceFunctionWithIdxNoGrad()

    batch_size = pred_points.shape[0]
    batch_hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    for b in range(batch_size):
        dist1, dist2, idx1, idx2 = _dist_with_idx(
            pred_points[b : b + 1], gt_points[b : b + 1]
        )
        gt_seg_aligned = torch.gather(gt_seg[b : b + 1], 1, idx1.long())

        gt_np = gt_seg_aligned[0].cpu().numpy()
        pred_np = pred_seg[b].cpu().numpy()

        pred_np = pred_np.astype(np.int64)
        gt_np = gt_np.astype(np.int64)

        gt_np[gt_np == ignore_index] = -1
        pred_np[gt_np == -1] = -1

        hist = fast_hist(pred_np, gt_np, num_classes)
        batch_hist += hist

    return batch_hist
