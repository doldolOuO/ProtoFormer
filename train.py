import warnings, time, random
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from dataloader import get_dataloader
from config import params
from models.ProtoFormer import ProtoFormer
from cuda.ChamferDistance import L1_ChamferDistance
from loss.ssc_loss import Loss_train


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        # torch.backends.cudnn.deterministic = True


def main():
    warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")
    # default setting
    cfg = params()
    DATASET = cfg.dataset
    MODEL = 'ProtoFormer'
    FLAG = 'train'
    BATCH_SIZE = int(cfg.batch_size)
    best_loss = 99999
    resume_epoch = 1

    # create log
    TIME_FLAG = time.asctime(time.localtime(time.time()))
    log_dir = f'./log/{MODEL}_{BATCH_SIZE}_{DATASET}_{FLAG}_{TIME_FLAG}'
    if not os.path.exists(os.path.join(log_dir)):
        os.makedirs(os.path.join(log_dir))

    # loss function
    loss_cd = L1_ChamferDistance()

    # dataset loading
    if DATASET == 'SSC-PC':
        root_dir = '/data/FangChengHao/data/SSC-PC'
        train_list_file = './dataset/train_ssc_pc_list.txt'
        test_list_file = './dataset/test_ssc_pc_list.txt'
        model = ProtoFormer(16,1024,2048,1.2,[1, 2])
        loss_train = Loss_train(seg_weight=[1.50, 0.96, 1.03, 1.10, 1.67, 1.10, 1.14, 1.74,
                                            0.69, 1.06, 2.09, 1.10, 1.06, 1.57, 1.68, 0.69])
    else:
        root_dir = '/data/FangChengHao/data/NYUCAD-PC'
        train_list_file = './dataset/train_nyucad_pc_list.txt'
        test_list_file = './dataset/test_nyucad_pc_list.txt'
        model = ProtoFormer(12, 1024, 1024, 1, [2, 2, 2])
        loss_train = Loss_train(seg_weight=[0, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1])

    # create models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    model.to(device)

    train_loader = get_dataloader(root_dir,
                                  train_list_file,
                                  "train",
                                  BATCH_SIZE,
                                  True,
                                  cfg.nThreads)
    test_loader = get_dataloader(root_dir,
                                 test_list_file,
                                 "valid",
                                 BATCH_SIZE,
                                 False,
                                 cfg.nThreads)

    # optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.9)

    # saving hyperparameters
    CONFIG_FILE = f'./log/{MODEL}_{BATCH_SIZE}_{DATASET}_{FLAG}_{TIME_FLAG}/CONFIG.txt'
    with open(CONFIG_FILE, 'w') as f:
        f.write('RESUME:' + str(cfg.resume) + '\n')
        f.write('DATASET:' + str(DATASET) + '\n')
        f.write('FLAG:' + str(FLAG) + '\n')
        f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
        f.write('MAX_EPOCH:' + str(int(cfg.n_epochs)) + '\n')
        f.write(str(cfg.__dict__))

    # models loading
    if cfg.resume:
        ckpt_dict = torch.load(cfg.ckpt)
        model.load_state_dict(ckpt_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt_dict['scheduler_state_dict'])
        resume_epoch = ckpt_dict['epoch'] + 1
        best_loss = ckpt_dict['loss']
        scheduler.step()

    # training
    set_seed()
    for epoch in range(resume_epoch, cfg.n_epochs + 1):
        model.train()
        n_batches = len(train_loader)
        with tqdm(train_loader) as t:
            for batch_idx, data in enumerate(t):
                partial = data[0].to(device)
                gt = data[1].to(device)
                label = data[2].to(device)
                out = model(partial)
                loss_out = loss_train(out, torch.cat((gt, label.unsqueeze(-1)), dim=-1))
                loss = loss_out['sum_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch,
                                                                  cfg.n_epochs,
                                                                  batch_idx + 1,
                                                                  n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [1e3 * loss_out['last_cd'].data.cpu(),
                                                                1e1 * loss_out['last_seg'].data.cpu()
                                                                ]])
        if epoch % int(cfg.eval_epoch) == 0:
            with torch.no_grad():
                model.eval()
                Loss = 0
                with tqdm(test_loader) as t:
                    for batch_idx, data in enumerate(t):
                        partial = data[0].to(device)
                        gt = data[1].to(device)
                        out = model(partial)
                        loss = loss_cd(out[-1][:, :, :3].contiguous(), gt)
                        Loss += loss * 1e3
                    Loss = Loss / len(test_loader)
                    if Loss < best_loss:
                        best_loss = Loss
                        best_epoch = epoch
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': Loss},
                            f'./log/{MODEL}_{BATCH_SIZE}_{DATASET}_{FLAG}_{TIME_FLAG}/ckpt_{epoch}_{Loss}.pt')
                        print('best epoch: ', best_epoch, 'cd: ', best_loss.item())
                    print(epoch, ' ', Loss.item(),
                          'lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

if __name__ == '__main__':
    main()
