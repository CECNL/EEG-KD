from scipy import io
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np


path = './BCICIV_2a/'

def split_train_valid_set(x_train, y_train, ratio):
    s = y_train.argsort()
    x_train = x_train[s]
    y_train = y_train[s]

    cL = int(len(x_train) / 4)

    class1_x = x_train[ 0 * cL : 1 * cL ]
    class2_x = x_train[ 1 * cL : 2 * cL ]
    class3_x = x_train[ 2 * cL : 3 * cL ]
    class4_x = x_train[ 3 * cL : 4 * cL ]

    class1_y = y_train[ 0 * cL : 1 * cL ]
    class2_y = y_train[ 1 * cL : 2 * cL ]
    class3_y = y_train[ 2 * cL : 3 * cL ]
    class4_y = y_train[ 3 * cL : 4 * cL ]

    vL = int(len(class1_x) / ratio)

    x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
    y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

    x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
    y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

    return x_train, y_train, x_valid, y_valid


def append_other_subject(x_train, y_train, x_valid, y_valid, subject, ratio):
    subject_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']

    for otherSubject in subject_list:
        if otherSubject == subject:
            continue
        else:
            data = io.loadmat(path + 'BCIC_' + otherSubject + '_T.mat')
            x = torch.Tensor(data['x_train']).unsqueeze(1)
            y = torch.Tensor(data['y_train']).view(-1)

            xt, yt, xv, yv = split_train_valid_set(x, y, ratio)

            x_train = torch.cat((x_train, xt))
            y_train = torch.cat((y_train, yt))
            x_valid = torch.cat((x_valid, xv))
            y_valid = torch.cat((y_valid, yv))

    return x_train, y_train, x_valid, y_valid



def get_dl(subject, ratio=8, shuffle=True, opt=None):
    batch_size = opt.batch_size
    train = io.loadmat(path + 'BCIC_' + subject + '_T.mat')
    test = io.loadmat(path + 'BCIC_' + subject + '_E.mat')

    x_train = torch.Tensor(train['x_train']).unsqueeze(1)
    y_train = torch.Tensor(train['y_train']).view(-1)
    x_test = torch.Tensor(test['x_test']).unsqueeze(1)
    y_test = torch.Tensor(test['y_test']).view(-1)

    x_train, y_train, x_valid, y_valid = split_train_valid_set(x_train, y_train, ratio=ratio)

    if opt.scheme == 'dependent':
        x_train, y_train, x_valid, y_valid = append_other_subject(x_train, y_train, x_valid, y_valid, subject, ratio)
    
    if opt.scheme == 'independent':
        L = len(x_train)
        vL = len(x_valid)

        x_train, y_train, x_valid, y_valid = append_other_subject(x_train, y_train, x_valid, y_valid, subject, ratio)
        x_train = x_train[L:]
        y_train = y_train[L:]
        x_valid = x_valid[vL:]
        y_valid = y_valid[vL:]

    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.long().to(dev)
    
    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    test_dataset = Data.TensorDataset(x_test, y_test)

    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 0,
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 36,
        shuffle = False,
        num_workers = 0,
    )
    testloader =  Data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
    )

    return trainloader, validloader, testloader