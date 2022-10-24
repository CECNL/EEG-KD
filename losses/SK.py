import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn


class SK(nn.Module):
    def __init__(self):
        super(SK, self).__init__()
        self.mse = nn.MSELoss()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, g_s, g_t):
        return sum([self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.size(0)
        c_s = f_s.size(1)
        c_t = f_t.size(1)

        # zero-centered
        f_s = f_s - f_s.mean(axis=3).unsqueeze(axis=3)
        f_t = f_t - f_t.mean(axis=3).unsqueeze(axis=3)

        # pair-wise zero-centered cosine similarity
        f_s = nn.functional.normalize(f_s, dim=3)
        f_t = nn.functional.normalize(f_t, dim=3)
        f_s = f_s.flatten(start_dim=1)
        f_t = f_t.flatten(start_dim=1)
        G_s = torch.mm(f_s, torch.t(f_s)) / c_s
        G_t = torch.mm(f_t, torch.t(f_t)) / c_t

        return self.mse(G_t, G_s)