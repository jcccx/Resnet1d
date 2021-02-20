from torch.utils.data import DataLoader
from tqdm import tqdm
from ResE import ResEncoder, BasicBlock, MyDataset
from sklearn.model_selection import train_test_split
import torch
import os
import pickle
import numpy as np
from torch.utils import data
import torch.utils.data as Data
from torch import nn, optim

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time

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
    loss = torch.nn.CrossEntropyLoss()
    #for _ in tqdm(range(num_epochs), desc="epoch", leave=False):
    for epoch in range(num_epochs):

        #print(_)
        train_ls_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            #print("y_hat:", y_hat.shape)
            #print("y:", y.shape)
            ls = loss(y_hat, y)

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            train_ls_sum += ls.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_ls_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def dataset(root):
    users = os.listdir(root)
    users_path = [os.path.join(root, user) for user in users]
    user_apps = [os.listdir(user_path) for user_path in users_path]
    # print(user_apps)
    app_tlabel = []
    app_label1 = []
    datas = []
    labels = []
    for apps in user_apps:
        for app in apps:
            # print(app)
            apptag = app[11:-5]
            app_path = os.path.join(root, app[:10] + app[-5:], app, app[:-5] + ".pkl")
            # print(app_path)
            fp = open(app_path, 'rb')
            if fp:
                # label
                if apptag not in app_tlabel:
                    app_tlabel.append(apptag)
                # label = torch.tensor([int(app_tlabel.index(apptag)), int(len(user_tlabel) - 1)]
                # feature
                while True:
                    try:
                        cc = pickle.load(fp)
                        # print(aa)
                        datas.append(np.array(cc).T)
                        labels.append(int(app_tlabel.index(apptag)))
                    except EOFError:

                        break
            fp.close()
    for i in range(len(app_tlabel)):
        if (labels.count(i) == 1):
            app_label1.append(app_tlabel[i])
            del datas[labels.index(i)]
            labels.remove(i)
    #print(len(app_tlabel) - len(app_label1))
    print(len(labels))
    return datas,labels

def main():
    path = r"D:\exper\compiler\stage2_compiler\sc_chunk"
    datas, labels = dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=0,
                                                        stratify=labels)
    train_db = MyDataset(X_train, y_train)
    test_db = MyDataset(X_test, y_test)
    batch_size = 1024
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # stage1:286, stage2:226
    net = ResEncoder(BasicBlock, [9, 5, 3], 226, 74)
    lr, num_epochs = 0.003, 1500
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()