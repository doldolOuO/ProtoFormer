import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
from typing import Iterable


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, Iterable):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1, pred.size(2))
        target = target.view(-1, 1).long()

        logpt = nn.functional.log_softmax(pred, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
