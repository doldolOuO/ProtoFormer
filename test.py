import torch
from tqdm import tqdm
import numpy as np
from dataloader import get_dataloader
from models.ProtoFormer import ProtoFormer
from cuda.ChamferDistance import L1_ChamferDistance, L2_ChamferDistance, F1Score
from seg_eval import eval_points_segmentation


DATASET = "NYUCAD-PC" # SSC-PC or NYUCAD-PC

# dataset loading
if DATASET == 'SSC-PC':
    root_dir = '/data/FangChengHao/data/SSC-PC'
    test_list_file = './dataset/test_ssc_pc_list.txt'
    ckpt_dir = './ckpt/sscpc.pt'
    class_names = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'celling',
                   'chair', 'desk', 'door', 'floor', 'other',
                   'sink', 'sofa', 'table', 'toilet', 'tv', 'wall']
    ignore_index = -1
    model = ProtoFormer(16,1024,2048,1.2,[1,2])
else:
    root_dir = '/data/FangChengHao/data/NYUCAD-PC'
    test_list_file = './dataset/test_nyucad_pc_list.txt'
    ckpt_dir = './ckpt/nyucadpc.pt'
    class_names = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair',
        'bed', 'sofa', 'table', 'tvs', 'furn', 'objs']
    ignore_index = 0
    model = ProtoFormer(12, 1024, 1024, 1, [2, 2, 2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model)
model.to(device)
state_dict = torch.load(ckpt_dir)['model_state_dict']
model.load_state_dict(state_dict)

test_loader = get_dataloader(root_dir,
                             test_list_file,
                             "test",
                             1,
                             False,
                             8)

num_classes = len(class_names)
total_hist = np.zeros((num_classes, num_classes), dtype=np.int64)

loss_cd1 = L1_ChamferDistance()
loss_cd2 = L2_ChamferDistance()
loss_f1 = F1Score()

with torch.no_grad():
    model.eval()
    i = 0
    Loss1 = 0
    Loss2 = 0
    f1_final = 0
    for data in tqdm(test_loader):
        i += 1
        partial = data[0].to(device)
        gt = data[1].to(device)
        label = data[2].to(device)
        # if DATASET == 'NYUCAD-PC':
        #     label = label - 1
        out = model(partial)
        completion = out[-1][:, :, :3].contiguous()
        logits = out[-1][:, :, 3:].contiguous()
        preds = torch.argmax(logits, dim=-1)

        # Accumulate confusion matrix
        batch_hist = eval_points_segmentation(
            completion, gt, preds, label, num_classes, ignore_index
        )
        total_hist += batch_hist

        # Simultaneously compute point cloud completion loss and F1-score
        loss1 = loss_cd1(completion, gt)
        loss2 = loss_cd2(completion, gt)
        f1, _, _ = loss_f1(completion, gt)
        f1 = f1.mean()
        Loss1 += loss1.item()
        Loss2 += loss2.item()
        f1_final += f1.item()

# Calculate overall metrics
def per_class_iou(hist: np.ndarray) -> np.ndarray:
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / denominator
    iou = np.where(denominator == 0, np.nan, iou)
    return iou

def get_acc(hist: np.ndarray) -> float:
    total = hist.sum()
    correct = np.diag(hist).sum()
    return float(correct / total) if total > 0 else float('nan')

def get_acc_cls(hist: np.ndarray) -> float:
    denominator = hist.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / denominator
    acc_cls = np.where(denominator == 0, np.nan, acc_cls)
    return float(np.nanmean(acc_cls))

Loss1 = Loss1 / i
Loss2 = Loss2 / i
f1_final = f1_final / i

iou = per_class_iou(total_hist)
if 0 <= ignore_index < num_classes:
    iou[ignore_index] = np.nan
miou = np.nanmean(iou)
acc = get_acc(total_hist)
acc_cls = get_acc_cls(total_hist)

print(f"The CD L1 is: {Loss1 * 1e3:.3f}")
print(f"The CD L2 is: {Loss2 * 1e4:.3f}")
print(f"The F1-score is: {f1_final:.3f}")
print(f"mAcc is: {acc_cls * 1e2:.2f}")
print(f"mIoU is: {miou * 1e2:.2f}")