import time
import torch
from torch import nn
import torch.optim as optim
import gpdata
import os
import numpy as np
import gpdateutil
import gpfileutil

epochs = 5000
batches = 64
lr = 0.01
input_size = 45
output_size = 3
hidden_size = 100
train_or_predict = 'predict'
gpmc = '奥拓电子'
gpdm = '2587'



class FizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FizBuzNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        hidden = self.hidden(batch)
        activated = torch.sigmoid(hidden)
        out = self.out(activated)
        return out


if train_or_predict == 'train':
    daystartend = gpdateutil.getstart_end(str('2587').zfill(6))
    gpxyclass = gpdata.GetData('2587','奥拓电子',startdate=daystartend[0],enddate=daystartend[1])
    trX, trY, teX, teY = gpxyclass.get_pytorch_data()
    if torch.cuda.is_available():
        xtype = torch.cuda.FloatTensor
        ytype = torch.cuda.LongTensor
    else:
        xtype = torch.FloatTensor
        ytype = torch.LongTensor
    x = torch.from_numpy(trX).type(xtype)
    y = torch.from_numpy(trY).type(ytype)
    net = FizBuzNet(input_size, hidden_size, output_size)
    zdzql = gpfileutil.getmaxzql(gpdmmc=gpxyclass.gpdmmc, path='./data')
    if os.path.exists('./data/{}_{}_model.pth'.format(zdzql, gpxyclass.gpdmmc)):
        net.load_state_dict(torch.load('./data/{}_{}_model.pth'.format(zdzql, gpxyclass.gpdmmc)))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    total_time = []
    no_of_batches = int(len(trX) / batches)
    for epoch in range(epochs):
        for batch in range(no_of_batches):
            start = batch * batches
            end = start + batches
            x_ = x[start:end]
            y_ = y[start:end]
            start = time.time()
            hyp = net(x_)
            loss = loss_fn(hyp, y_)
            optimizer.zero_grad()
            loss.backward()
            total_time.append(time.time() - start)
            optimizer.step()
        if epoch % 10:
            print(epoch, loss.item())
    total_sum = sum(total_time)
    total_len = len(total_time)
    print(total_sum, total_len, total_sum / total_len)

    zql = 0 # zql 90以上再保存
    # Test
    with torch.no_grad():
        x = torch.from_numpy(teX).type(xtype)
        y = torch.from_numpy(teY).type(ytype)
        hyp = net(x)
        output = loss_fn(hyp, y)
        for i in range(len(teX)):
            if teY[i] != hyp[i].max(0)[1].item():
                print(
                    'Number: {} -- Actual: {} -- Prediction: {}'.format(
                        teX[i], teY[i], hyp[i].max(0)[1].item()))
        print('Test loss: {} -- Test_X_len: {}', output.item() / len(x), len(x))
        accuracy = hyp.max(1)[1] == y
        zql = accuracy.sum().item() / len(accuracy)
        print('accuracy: ', accuracy.sum().item() / len(accuracy))
    if zql > 0.7:
        torch.save(net.state_dict(), './data/{}_{}_model.pth'.format(round(zql, 4), gpxyclass.gpdmmc))

if train_or_predict == 'predict':
    gpxyclass = gpdata.GetData(gpdm, gpmc, tdsdate='20240312', tdedate='20240314')
    x_pr = gpxyclass.get_gp_x_predict()
    x = []
    for row in x_pr.itertuples():
        if len(row.zuhe) > 0:
            ret1 = [int(i) for i in str(row.zuhe).split(' ')]
            x.append(ret1)
    net = FizBuzNet(input_size, hidden_size, output_size)
    zdzql = gpfileutil.getmaxzql(gpxyclass.gpdmmc)
    if os.path.exists('./data/{}_{}_model.pth'.format(zdzql, gpxyclass.gpdmmc)):
        net.load_state_dict(torch.load('./data/{}_{}_model.pth'.format(zdzql, gpxyclass.gpdmmc)))
    else:
        print('先训练')
    np_x = np.array(x)
    xtype = torch.FloatTensor
    pr_x = torch.from_numpy(np_x).type(xtype)
    hyp = net(pr_x)
    ycz = hyp[0].max(0)[1].item()
    print(ycz)