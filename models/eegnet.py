import torch
import torch.nn as nn



class EEGNet22(nn.Module):
    def __init__(self, channel_num=22):
        super(EEGNet22, self).__init__()

        F1 = 8
        F2 = 16
        D = 2
        cn = channel_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, D*F1, (cn, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, 16), padding=(0, 8), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
        )

        self.dropout = nn.Dropout(0.5)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.elu = nn.ELU()
        self.classifier = nn.Linear(16*17, 4, bias=True)

    def forward(self, x, fea=False):
        f1 = x = self.conv1(x)
        x = self.conv2(x)
        f2 = x = self.elu(x)
        x = self.pool1(x)
        x = self.Conv3(x)
        f3 = x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(-1, 16*17)
        x = self.classifier(x)
        if fea:
            return x, [f1, f2, f3]
        return x



class EEGNet7(nn.Module):
    def __init__(self, channel_num=7):
        super(EEGNet7, self).__init__()

        F1 = 8
        F2 = 16
        D = 2
        cn = channel_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, D*F1, (cn, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, 16), padding=(0, 8), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
        )

        self.dropout = nn.Dropout(0.5)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.elu = nn.ELU()
        self.classifier = nn.Linear(16*17, 4, bias=True)

    def forward(self, x, fea=False):
        f1 = x = self.conv1(x)
        x = self.conv2(x)
        f2 = x = self.elu(x)
        x = self.pool1(x)
        x = self.Conv3(x)
        f3 = x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(-1, 16*17)
        x = self.classifier(x)
        if fea:
            return x, [f1, f2, f3]
        return x



class EEGNet4(nn.Module):
    def __init__(self, channel_num=4):
        super(EEGNet4, self).__init__()

        F1 = 8
        F2 = 16
        D = 2
        cn = channel_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, D*F1, (cn, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, 16), padding=(0, 8), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
        )

        self.dropout = nn.Dropout(0.5)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.elu = nn.ELU()
        self.classifier = nn.Linear(16*17, 4, bias=True)

    def forward(self, x, fea=False):
        f1 = x = self.conv1(x)
        x = self.conv2(x)
        f2 = x = self.elu(x)
        x = self.pool1(x)
        x = self.Conv3(x)
        f3 = x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(-1, 16*17)
        x = self.classifier(x)
        if fea:
            return x, [f1, f2, f3]
        return x