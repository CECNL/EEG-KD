import torch
import torch.nn as nn

class square_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2


class SCCNet22(nn.Module):
    def __init__(self):
        super(SCCNet22, self).__init__()
        # spatial 
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)

        # spatial-temporal
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)

        # square layer
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(840, 4, bias=True)

    def forward(self, x, fea=False):
        x = self.conv1(x)
        f1 = x = self.Bn1(x)
        x = self.conv2(x)
        f2 = x = self.Bn2(x)
        f3 = x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 840)
        x = self.classifier(x)

        if fea:
            return x, [f1, f2, f3]
        return x


class SCCNet7(nn.Module):
    def __init__(self):
        super(SCCNet7, self).__init__()

        # spatial 
        self.conv1 = nn.Conv2d(1, 22, (7, 1))
        self.Bn1 = nn.BatchNorm2d(22)

        # spatial-temporal
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)

        # square layer
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(840, 4, bias=True)

    def forward(self, x, fea=False):
        x = self.conv1(x)
        f1 = x = self.Bn1(x)
        x = self.conv2(x)
        f2 = x = self.Bn2(x)
        f3 = x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 840)
        x = self.classifier(x)

        if fea:
            return x, [f1, f2, f3]
        return x

class SCCNet4(nn.Module):
    def __init__(self):
        super(SCCNet4, self).__init__()


        # spatial 
        self.conv1 = nn.Conv2d(1, 22, (4, 1))
        self.Bn1 = nn.BatchNorm2d(22)

        # spatial-temporal
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)

        # square layer
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(840, 4, bias=True)

    def forward(self, x, fea=False):
        x = self.conv1(x)
        f1 = x = self.Bn1(x)
        x = self.conv2(x)
        f2 = x = self.Bn2(x)
        f3 = x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 840)
        x = self.classifier(x)
        if fea:
            return x, [f1, f2, f3]
        return x