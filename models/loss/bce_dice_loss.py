import torch
import torch.nn as nn
import torch.nn.functional as F

class BCE_DiceLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(BCE_DiceLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target, mask, reduce=True):
    #def forward(self, input, target, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()



        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        #d = (2 * a) / (b + c)
        dice_loss = 1 - (2 * a) / (b + c)

        criterion = nn.BCELoss()
        bce_loss = criterion(input, target)


        loss = self.weight * (bce_loss + dice_loss)

        #loss = self.weight * dice_loss

        if reduce:
            loss = torch.mean(loss)

        return loss