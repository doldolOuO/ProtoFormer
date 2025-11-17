class params:
    def __init__(self):
        #Training parameters
        self.batch_size = 8
        self.nThreads = 8
        self.lr = 0.001
        self.milestones = [150, 300, 450]
        self.dataset = 'SSC-PC' # SSC-PC or NYUCAD-PC
        self.n_epochs = 600
        self.eval_epoch = 1
        self.resume = False
        self.ckpt = "./ckpt/nyucadpc.pt"