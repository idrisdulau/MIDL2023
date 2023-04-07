import torch

class DiceLoss(torch.nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, yPred, yTrue):
        assert yPred.size() == yTrue.size()
        yPred = yPred[:, 0].contiguous().view(-1)
        yTrue = yTrue[:, 0].contiguous().view(-1)
        intersection = (yPred * yTrue).sum()
        dsc = (2. * intersection + self.smooth) / (yPred.sum() + yTrue.sum() + self.smooth)
        return 1. - dsc
