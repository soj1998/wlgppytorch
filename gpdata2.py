import akshare as ak
import pandas as pd
import numpy as np


def biaozhunhua(a):
    '''
    涨跌幅 +10 振幅 20以内 换手率 20以内
    '''
    if a < -10:
        return 0
    if -10 <= a < -5:
        return 1
    if -5 <= a < 0:
        return 2
    if 0 <= a < 5:
        return 3
    if 5 <= a < 10:
        return 4
    if 10 <= a < 20:
        return 5
    if a > 20:
        return 6


def binary_encoder(input_size):
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret

    return wrapper


def zuhe_encoder(input_size):
    def wrapper(ret):
        return ret[0] + ret[1] + ret[2]

    return wrapper


def training_test_gen(x, y):
    assert len(x) == len(y)
    indices = np.random.permutation(range(len(x)))
    split_size = int(0.9 * len(indices))
    trX = x[indices[:split_size]]
    trY = y[indices[:split_size]]
    teX = x[indices[split_size:]]
    teY = y[indices[split_size:]]
    return trX, trY, teX, teY


def get_pytorch_data(input_size=10, limit=1000):
    x = []
    y = []
    encoder = binary_encoder(input_size)
    for i in range(limit):
        x.append(encoder(i))
        if i % 15 == 0:
            y.append(0)
        elif i % 5 == 0:
            y.append(1)
        elif i % 3 == 0:
            y.append(2)
        else:
            y.append(3)
    return training_test_gen(np.array(x), np.array(y))


stock_zh_index_daily_df = pd.DataFrame()
try:
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str('2529').zfill(6), period="daily",
                                            start_date="20230101", end_date='20231231',
                                            adjust="")
    encoder = binary_encoder(5)
    zhencoder = zuhe_encoder(15)
    rql, spl, zdfl = [], [], []
    bjlist = pd.DataFrame()
    for row in stock_zh_a_hist_df.itertuples([['日期', '收盘', '涨跌幅']]):
        if len(rql) >= 4:
            inde = [i + 1 for i in range(4)]
            bjlist_x = pd.DataFrame()
            for j in inde:
                bjlist_x.at[0, 'rq_%d' % j] = rql[j - 1]
                bjlist_x.at[0, 'sp_%d' % j] = spl[j - 1]
                bjlist_x.at[0, 'zdf_%d' % j] = zdfl[j - 1]
            bjlist = pd.concat([bjlist, bjlist_x])
            rql.pop(0)
            spl.pop(0)
            zdfl.pop(0)
        rql.append(row.日期)
        spl.append(row.收盘)
        zdfl.append(row.涨跌幅)
        stock_zh_a_hist_df_shape = stock_zh_a_hist_df.shape
        if stock_zh_a_hist_df_shape[0] > 0 \
                and row.日期 == stock_zh_a_hist_df.iloc[stock_zh_a_hist_df_shape[0] - 1, 0]:
            bjlist_x1 = pd.DataFrame()
            for j in inde:
                bjlist_x1.at[0, 'rq_%d' % j] = rql[j - 1]
                bjlist_x1.at[0, 'sp_%d' % j] = spl[j - 1]
                bjlist_x1.at[0, 'zdf_%d' % j] = zdfl[j - 1]
            bjlist = pd.concat([bjlist, bjlist_x1])
    bjlist['y'] = bjlist.apply(lambda row1: 1 if (row1['sp_4'] - row1['sp_1']) / row1['sp_1'] >= 0.05
        and row1['zdf_2'] + row1['zdf_3'] + row1['zdf_4'] > 3
            and row1['zdf_2'] > -1 and row1['zdf_3'] > -1 and row1['zdf_4'] > -1 else 0, axis=1)
    bjlist_x = stock_zh_a_hist_df
    bjlist_x['zhenfu'] = bjlist_x.apply(lambda row2: encoder(round(float(row2.振幅))), axis=1)
    bjlist_x['zhangdiefu'] = bjlist_x.apply(lambda row3: encoder(round(float(row3.涨跌幅) + 10)), axis=1)
    bjlist_x['huanshoulv'] = bjlist_x.apply(lambda row2: encoder(round(float(row2.换手率))), axis=1)
    bjlist_x['zuhe'] = bjlist_x.apply(lambda row2:
                                      zhencoder([encoder(round(float(row2.振幅))), encoder(round(float(row2.涨跌幅) + 10)),
                                                 encoder(round(float(row2.换手率)))]), axis=1)
    rqll, zuhel = [], []
    bjlist_x1 = pd.DataFrame()
    for row in bjlist_x.itertuples([['日期', 'zuhe']]):
        if len(rqll) >= 3:
            inde = [i + 1 for i in range(3)]
            bjlist_xx = pd.DataFrame()
            for j in inde:
                zuhelist = zuhel[j - 1]
                bjlist_xx.at[0, 'rq_%d' % j] = rqll[j - 1]
                bjlist_xx.at[0, 'zuhe_%d' % j] = ' '.join([str(x) for x in zuhelist])
            bjlist_x1 = pd.concat([bjlist_x1, bjlist_xx])
            rqll.pop(0)
            zuhel.pop(0)
        rqll.append(row.日期)
        zuhel.append(row.zuhe)
        bjlist_x_shape = bjlist_x.shape
        if bjlist_x_shape[0] > 0 \
                and row.日期 == bjlist_x.iloc[bjlist_x_shape[0] - 1, 0]:
            bjlist_xx1 = pd.DataFrame()
            for j in inde:
                zuhelist = zuhel[j - 1]
                bjlist_xx1.at[0, 'rq_%d' % j] = rqll[j - 1]
                bjlist_xx1.at[0, 'zuhe_%d' % j] = ' '.join([str(x) for x in zuhelist])
            bjlist_x1 = pd.concat([bjlist_x1, bjlist_xx1])
    bjlist_x1['rq'] = bjlist_x1['rq_3']
    bjlist_x1['zuhe'] = bjlist_x1.apply(lambda row2: row2.zuhe_1 + ' ' + row2.zuhe_2 + ' ' + row2.zuhe_3, axis=1)
    bjlist_x3 = pd.DataFrame()
    for row in bjlist_x.itertuples():
        bjlist_x3_1 = {'日期': [row.日期], '开盘': [row.开盘], '收盘': [row.收盘], '最高': [row.最高],
                       '最低': [row.最低], '成交量': [row.成交量], '成交额': [row.成交额], '振幅': [row.振幅],
                       '涨跌幅': [row.涨跌幅], '涨跌额': [row.涨跌额], '换手率': [row.换手率], 'zhenfu': [row.zhenfu],
                       'zhangdiefu': [row.zhangdiefu], 'huanshoulv': [row.huanshoulv], 'zuhe': ['']}
        for rowx in bjlist_x1.itertuples():
            if rowx.rq == row.日期:
                bjlist_x3_1['zuhe'] = [rowx.zuhe]
        bjlist_x3_1['y'] = -1
        for rowy in bjlist.itertuples():
            if rowy.rq_1 == row.日期:
                bjlist_x3_1['y'] = [rowy.y]
        bjlist_x3_2 = pd.DataFrame(bjlist_x3_1)
        bjlist_x3 = pd.concat([bjlist_x3, bjlist_x3_2])
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
    print(1)
except ValueError:
    print('error', ValueError)
