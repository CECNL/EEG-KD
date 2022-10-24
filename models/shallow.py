import torch
import torch.nn as nn


class square_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2


class ShallowConvNet22(nn.Module):
    def __init__(self):
        super(ShallowConvNet22, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (22, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*74, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x, fea=False):
        f1 = x = self.conv1(x)
        x = self.conv2(x)
        f2 = x = self.Bn1(x)
        f3 = x = self.SquareLayer(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*74)
        x = self.classifier(x)

        if fea:
            return x, [f1, f2, f3]
        return x


class ShallowConvNet7(nn.Module):
    def __init__(self):
        super(ShallowConvNet7, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (7, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*74, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x, fea=False):
        f1 = x = self.conv1(x)
        x = self.conv2(x)
        f2 = x = self.Bn1(x)
        f3 = x = self.SquareLayer(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*74)
        x = self.classifier(x)
        
        if fea:
            return x, [f1, f2, f3]
        return x

class ShallowConvNet4(nn.Module):
    def __init__(self):
        super(ShallowConvNet4, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (4, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*74, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x, fea=False):
        f1 = x = self.conv1(x)
        x = self.conv2(x)
        f2 = x = self.Bn1(x)
        f3 = x = self.SquareLayer(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*74)
        x = self.classifier(x)
        
        if fea:
            return x, [f1, f2, f3]
        return x