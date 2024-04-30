import torch
import gpdata
import os
import gpfileutil
import netclass as mynet
import numpy as np


def predict(input_size, output_size, hidden_size, gpmc, gpdm, p_tdsdate, p_tdedate):
    input_size = input_size
    output_size = output_size
    hidden_size = hidden_size
    gpmc = gpmc
    gpdm = gpdm
    gpxyclass = gpdata.GetData(gpdm, gpmc, tdsdate=p_tdsdate, tdedate=p_tdedate)
    x_pr = gpxyclass.get_gp_x_predict()
    x = []
    for row in x_pr.itertuples():
        if len(row.zuhe) > 0:
            ret1 = [int(i) for i in str(row.zuhe).split(' ')]
            x.append(ret1)
    net = mynet.FizBuzNet(input_size, hidden_size, output_size)
    zdzql = gpfileutil.getmaxzql(gpxyclass.gpdmmc)
    if os.path.exists('./data/{}_{}_model.pth'.format(gpxyclass.gpdmmc, zdzql)):
        net.load_state_dict(torch.load('./data/{}_{}_model.pth'.format(gpxyclass.gpdmmc, zdzql)))
    else:
        print('先训练')
        raise ValueError("先训练")
    np_x = np.array(x)
    xtype = torch.FloatTensor
    pr_x = torch.from_numpy(np_x).type(xtype)
    hyp = net(pr_x)
    ycz = hyp[0].max(0)[1].item()
    print(gpmc, gpdm, zdzql, ycz)