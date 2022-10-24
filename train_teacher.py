import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import random

from models import model_dict
from dataloader import get_dl
from loops import train_vanilla, test

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # setting
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    parser.add_argument('--repeat', type=int, default=10, help='repeat times')

    # model
    parser.add_argument('--model', type=str, default='SCCNet22',
                        choices=['SCCNet22', 'SCCNet7', 'SCCNet4', 
                                 'EEGNet22', 'EEGNet7', 'EEGNet4',
                                 'Shallow22', 'Shallow7', 'Shallow4'])
    parser.add_argument('--scheme', type=str, default='individual',
                        choices=['individual', 'dependent', 'independent'])
    parser.add_argument('--save_folder', type=str, default='./savedata/teacher', help='folder to save teacher model')


    opt = parser.parse_args()

    opt.subject_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']

    if opt.model[-1] == '7':
        opt.channel = [6, 7, 8, 9, 10, 11, 12]
    elif opt.model[-1] == '4':
        opt.channel = [0, 6, 12, 21]
    else:
        opt.channel = [i for i in range(22)]
    
    return opt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    opt = parse_option()

    record = np.zeros((len(opt.subject_list), opt.repeat))

    for s, sub in enumerate(opt.subject_list):
        best_acc = 0
        for i in range(opt.repeat):
            set_seed(i)
            train_dl, valid_dl, test_dl = get_dl(sub, opt=opt)
            net = model_dict[opt.model]()
            train_vanilla(net, train_dl, valid_dl, opt)
            test_acc = test(net, test_dl, opt)
            if best_acc < test_acc:
                best_acc = test_acc
                torch.save(net.state_dict(), os.path.join(opt.save_folder, sub+'.pt'))
            print(f"{sub}'s accuracy at {i+1} time: {test_acc}")
            record[s, i] = test_acc

        print(f"{sub}'s best accuracy: {best_acc}")
    np.save(os.path.join(opt.save_folder, 'record'), record)

if __name__ == '__main__':
    main()