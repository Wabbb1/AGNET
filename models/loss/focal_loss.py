import torch
import torch.nn as nn
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=1, alpha=0.5, beta=0.5, smooth=1, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma


    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        input = input * mask
        target = target * mask

        tp = torch.sum(input * target, dim=1)
        fp = torch.sum((1 - target) * input, dim=1)
        fn = torch.sum(target * (1 - input), dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        loss = 1 - tversky ** self.gamma

        loss = self.weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss