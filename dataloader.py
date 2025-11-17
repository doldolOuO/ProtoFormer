import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PCSSCDataset(Dataset):
    def __init__(self, root_dir, list_file, status):
        """
        Args:
            root_dir (str): dataset root
            list_file (str): sample list root
            status (str): "train", "valid" or "test"
        """
        self.root_dir = root_dir
        self.status = status
        with open(list_file, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]
        print(f'data num: {len(self.samples)}')

        if status in ["train", "valid"]:
            self.inputs = []
            self.label_points = []
            self.label_classes = []

            for sample in self.samples:
                sample_folder = os.path.join(self.root_dir, sample)
                points_path = sample_folder + "_input.npy"
                labels_path = sample_folder + "_gt.npy"

                points = np.load(points_path)  # (N, 3)
                labels = np.load(labels_path)  # (N, 4)

                lp = labels[:, :3]                  # (N, 3)
                lc = labels[:, 3].astype(np.int64)  # (N, )

                self.inputs.append(torch.from_numpy(points).float())
                self.label_points.append(torch.from_numpy(lp).float())
                self.label_classes.append(torch.from_numpy(lc).long())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.status in ["train", "valid"]:
            return self.inputs[idx], self.label_points[idx], self.label_classes[idx]
        else:  # status == "test"
            sample_folder = os.path.join(self.root_dir, self.samples[idx])
            points_path = sample_folder + "_input.npy"
            labels_path = sample_folder + "_gt.npy"

            points = np.load(points_path)  # (N, 3)
            labels = np.load(labels_path)  # (N, 4)

            label_points = labels[:, :3]                    # (N, 3)
            label_classes = labels[:, 3].astype(np.int64)   # (N, )

            input_points = torch.from_numpy(points).float()        # (N,3)
            label_points = torch.from_numpy(label_points).float()  # (N,3)
            label_classes = torch.from_numpy(label_classes).long() # (N,)

            return input_points, label_points, label_classes, sample_folder


def get_dataloader(root_dir, list_file, status, batch_size, shuffle, num_workers=0):
    dataset = PCSSCDataset(root_dir, list_file, status)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=False)
    return loader


if __name__ == "__main__":
    from tqdm import tqdm
    nyucad_pc_train_loader = get_dataloader('/data/FangChengHao/data/NYUCAD-PC',
                                  './dataset/train_nyucad_pc_list.txt',
                                  'train',
                                  1,
                                  True,
                                  8)
    nyucad_pc_test_loader = get_dataloader('/data/FangChengHao/data/NYUCAD-PC',
                                  './dataset/test_nyucad_pc_list.txt',
                                  'valid',
                                  1,
                                  False,
                                  8)
    ssc_pc_train_loader = get_dataloader('/data/FangChengHao/data/SSC-PC',
                                      './dataset/train_ssc_pc_list.txt',
                                      'train',
                                      1,
                                      True,
                                      8)
    ssc_pc_test_loader = get_dataloader('/data/FangChengHao/data/SSC-PC',
                                 './dataset/test_ssc_pc_list.txt',
                                 'valid',
                                 1,
                                 False,
                                 8)
    for data in tqdm(nyucad_pc_train_loader):
        partial = data[0].to('cuda')
        gt = data[1].to('cuda')
        label = data[2].to('cuda')
        print(partial.shape)
        print(gt.shape)
        print(label.shape)
