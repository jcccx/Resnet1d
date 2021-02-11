from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet1d import ResNet1D, MyDataset
import torch
from torch.utils import data
import torch.utils.data as Data
from torch import nn, optim
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import time
import pickle
import pandas as pd
import numpy as np


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss = torch.nn.CrossEntropyLoss()
    #for _ in tqdm(range(num_epochs), desc="epoch", leave=False):
    for epoch in range(num_epochs):
        #print(_)
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            #print(y_hat.shape)
            #print(y.shape)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step(epoch)
        test_acc = evaluate_accuracy(test_iter, net)
        print('\n epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch+ 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def main():
    path = r"D:\exper\compiler\stage2_compiler\sc_chunk"
    # path = r'D:/' #目录：eg："D:\exper\compiler\stage2_compiler\sc_chunk"
    train_db = MyDataset(path)
    batch_size = 64
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    train_db, test_db = torch.utils.data.random_split(train_db, [int(len(train_db)-1000), 1000])
    #print(len(train_db))
    #print(len(test_db))
    train_iter = torch.utils.data.DataLoader(train_db,
                           batch_size=batch_size,
                            shuffle=True,
                           num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_db,
                            batch_size=batch_size,
                           shuffle=False,
                            num_workers=num_workers)
    # (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
    # 34 layer (16*2+2): 16, 2, 4
    # 98 layer (48*2+2): 48, 6, 12
    net = ResNet1D(
        in_channels=74,
        base_filters=64,
        kernel_size=4,
        stride=2,
        groups=32,
        n_block=48,
        n_classes=226,  # stage1:286, stage2:226
        downsample_gap=6,
        increasefilter_gap=12,
        use_do=True)
    lr, num_epochs = 0.003, 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=1e-3)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()