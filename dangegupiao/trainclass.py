import time
import torch
from torch import nn
import torch.optim as optim
import gpdata
import os
import gpdateutil
import gpfileutil
import netclass as mynet


def train(epochs, batches, lr, input_size, output_size, hidden_size, gpmc, gpdm):
    epochs = epochs
    batches = batches
    lr = lr
    input_size = input_size
    output_size = output_size
    hidden_size = hidden_size
    gpmc = gpmc
    gpdm = gpdm
    daystartend = gpdateutil.getstart_end(str(gpdm).zfill(6))
    gpxyclass = gpdata.GetData(gpdm, gpmc, startdate=daystartend[0], enddate=daystartend[1])
    trX, trY, teX, teY, zdybl = gpxyclass.get_pytorch_data()
    if zdybl < 0.009:
        print('最大y比例太低了，才', str(zdybl))
        return
    if torch.cuda.is_available():
        xtype = torch.cuda.FloatTensor
        ytype = torch.cuda.LongTensor
    else:
        xtype = torch.FloatTensor
        ytype = torch.LongTensor
    x = torch.from_numpy(trX).type(xtype)
    y = torch.from_numpy(trY).type(ytype)
    net = mynet.FizBuzNet(input_size, hidden_size, output_size)
    zdzql = gpfileutil.getmaxzql(gpdmmc=gpxyclass.gpdmmc, path='./data')
    if os.path.exists('./data/{}_{}_model.pth'.format(gpxyclass.gpdmmc, zdzql)):
        print('读入模型', '{}_{}_model.pth'.format(zdzql, gpxyclass.gpdmmc))
        net.load_state_dict(torch.load('./data/{}_{}_model.pth'.format(gpxyclass.gpdmmc, zdzql)))
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

    zql = 0  # zql 90以上再保存
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
    if zql > zdzql:
        torch.save(net.state_dict(), './data/{}_{}_model.pth'.format(gpxyclass.gpdmmc, round(zql, 4)))
    else:
        print('{}_{}_model.pth'.format(gpxyclass.gpdmmc, zdzql), '准确率比{}高，不保存'.format(zql))
