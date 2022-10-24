import torch
import torch.nn as nn
import torch.optim as optim
from losses import KD


def train_vanilla(net, train_dl, valid_dl, opt):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(dev)
    CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    best_valid_loss = 1e10

    channel = opt.channel

    for epoch in range(opt.epochs):
        net.train()
        for xb, yb in train_dl:
            # subset along electrode subset
            xb = xb[:,:, channel]
            pred = net(xb)
            loss = CE(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # checkpoint : save network at epoch where lowest validation loss occurs
        net.eval()
        valid_loss = 0
        for xb, yb in valid_dl:
            with torch.no_grad():
                xb = xb[:,:, channel]
                pred = net(xb)
                valid_loss += CE(pred, yb).item()
                
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(net, './temp.pt')

    net = torch.load('./temp.pt').to(dev)
    return net



def train_KD(net, trainable_list, teacher, train_dl, valid_dl, criterion, opt):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(dev)
    trainable_list.append(net)

    teacher = teacher.to(dev)
    teacher.train()
    for param in teacher.parameters():
        param.requires_grad = False

    CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_list.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    best_valid_loss = 1e10

    channel = opt.channel
    for epoch in range(opt.epochs):
        net.train()
        for xb, yb in train_dl:
            target, fea_t = teacher(xb, fea=True)
            pred, fea_s = net(xb[:,:, channel], fea=True) # subset of electrode

            loss =  (1-opt.alpha) * CE(pred, yb) + opt.alpha * KD(pred, target, opt.T)
            loss += opt.beta * criterion(fea_s[1:], fea_t[1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # checkpoint : save network at epoch where lowest validation loss occurs
        net.eval()
        valid_loss = 0
        for xb, yb in valid_dl:
            with torch.no_grad():
                target, fea_t = teacher(xb, fea=True)
                pred, fea_s = net(xb[:,:, channel], fea=True)

                valid_loss += (1-opt.alpha) * CE(pred, yb) + opt.alpha * KD(pred, target, opt.T)
                valid_loss += opt.beta * criterion(fea_s[1:], fea_t[1:])
        

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(net, './temp.pt')

    net = torch.load('./temp.pt').to(dev)
    return net  


def test(net, test_dl, opt=None):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)

    for xb, yb in test_dl:
        with torch.no_grad():
            xb = xb[:,:, opt.channel]
            pred = net(xb)
            if torch.argmax(softmax(pred)) == yb:
                acc += 1
    return acc / len(test_dl)

