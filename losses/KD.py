import torch.nn as nn
import torch.nn.functional as F

def KD(pred, target, T):
    KL = nn.KLDivLoss(reduction='batchmean')
    loss = KL(F.log_softmax(pred/T, dim=1), F.softmax(target/T, dim=1)) * (T * T)
    return loss