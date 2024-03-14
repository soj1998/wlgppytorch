import akshare as ak
import pandas as pd
import os
import numpy as np
import re
import datetime


class GetData:
    def __init__(self, gpdm, gpmc, startdate='20230101',
                 enddate='20231231', tdsdate='20230101', tdedate='20230204'):
        self.gpdm = str(gpdm).zfill(6)
        self.gpmc = gpmc
        self.gpdmmc = self.gpdm + gpmc
        self.startdate = startdate
        self.enddate = enddate
        self.tdsdate = tdsdate
        self.tdedate = tdedate

    def binary_encoder(self, input_size):
        def wrapper(num):
            ret = [int(i) for i in '{0:b}'.format(num)]
            return [0] * (input_size - len(ret)) + ret

        return wrapper

    def zuhe_encoder(self):
        def wrapper(ret):
            return ret[0] + ret[1] + ret[2]
        return wrapper

    def __getitem__(self, item):
        if item == 'x':
            return 'x'
        if item == 'y':
            return 'y'

    def get_gp_akdata(self, xlorcs='train'):
        stock_zh_a_hist_df = pd.DataFrame()
        stdate = self.startdate
        endate = self.enddate
        if xlorcs == 'predict':
            stdate = self.tdsdate
            endate = self.tdedate
        try:
            stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str(self.gpdm).zfill(6), period="daily",
                                                    start_date=stdate, end_date=endate,
                                                    adjust="")
            # stock_zh_a_hist_df.to_excel('./data/%s.xlsx' % self.gpdmmc)
        except ValueError:
            print('error', ValueError)
        return stock_zh_a_hist_df

    def load_gp_xlsdata(self):
        stock_zh_a_hist_df = pd.DataFrame()
        if not os.path.exists('./data'):
            os.mkdir('./data')
        if not os.path.exists('./data/%s.xlsx' % self.gpdmmc):
            d1 = self.get_gp_akdata()
            bjlist_x = d1
            encoder = self.binary_encoder(5)
            zhencoder = self.zuhe_encoder()
            bjlist_x['zhenfu'] = bjlist_x.apply(lambda row2:
            encoder(round(30 if float(row2.振幅) > 30 else float(row2.振幅))), axis=1)
            bjlist_x['zhangdiefu'] = bjlist_x.apply(lambda row3:
            encoder(round(10 if float(row3.涨跌幅) > 10 else 0 if float(row3.涨跌幅) < -10 else
            float(row3.涨跌幅) +20)), axis=1)
            bjlist_x['huanshoulv'] = bjlist_x.apply(lambda row2:
            encoder(round(30 if float(row2.换手率) > 30 else float(row2.换手率))), axis=1)
            bjlist_x['zuhe'] = bjlist_x.apply(lambda row2:
                zhencoder([encoder(round(30 if float(row2.振幅) > 30 else float(row2.振幅))),
                    encoder(round(10 if float(row2.涨跌幅) > 10 else 0 if float(row2.涨跌幅) < -10 else
                           float(row2.涨跌幅) + 20)),
                    encoder(round(30 if float(row2.换手率) > 30 else float(row2.换手率)))]), axis=1)
            bjlist_x['日期'] = bjlist_x.apply(lambda row2:
                                              datetime.datetime.strptime(str(row2.日期), '%Y-%m-%d'), axis=1)
            stock_zh_a_hist_df = bjlist_x
            stock_zh_a_hist_df.to_excel('./data/%s.xlsx' % self.gpdmmc)
            return stock_zh_a_hist_df
        try:
            stock_zh_a_hist_df = pd.read_excel('./data/%s.xlsx' % self.gpdmmc)
        except ValueError:
            print('error', ValueError)
        return stock_zh_a_hist_df

    def get_gp_y(self):  # y不涉及到取新的来预测的问题
        ysdata = self.load_gp_xlsdata()
        rql, spl, zdfl = [], [], []
        bjlist = pd.DataFrame()
        for row in ysdata.itertuples([['日期', '收盘', '涨跌幅']]):
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
            stock_zh_a_hist_df_shape = ysdata.shape
            if stock_zh_a_hist_df_shape[0] > 0 \
                    and row.日期 == ysdata.loc[stock_zh_a_hist_df_shape[0] - 1, '日期']:
                bjlist_x1 = pd.DataFrame()
                for j in inde:
                    bjlist_x1.at[0, 'rq_%d' % j] = rql[j - 1]
                    bjlist_x1.at[0, 'sp_%d' % j] = spl[j - 1]
                    bjlist_x1.at[0, 'zdf_%d' % j] = zdfl[j - 1]
                bjlist = pd.concat([bjlist, bjlist_x1])
        bjlist['y'] = bjlist.apply(lambda row1: 1
        if 0.05 > (row1['sp_4'] - row1['sp_1']) / row1['sp_1'] >= 0.03
        else 2 if (row1['sp_4'] - row1['sp_1']) / row1['sp_1'] >= 0.05
                      and row1['zdf_2'] + row1['zdf_3'] + row1['zdf_4'] > 3
                      and row1['zdf_2'] > -1 and row1['zdf_3'] > -1 and row1[
                          'zdf_4'] > -1 else 0, axis=1)
        bjlist['y1'] = bjlist.apply(lambda row1: 1 if (row1['sp_4'] - row1['sp_1']) / row1['sp_1'] >= 0.03
        else 0, axis=1)
        return bjlist

    def get_gp_x(self):
        bjlist_x = self.load_gp_xlsdata()
        if bjlist_x.shape[0] == 0:
            return pd.DataFrame()
        rqll, zuhel = [], []
        bjlist_x1 = pd.DataFrame()
        for row in bjlist_x.itertuples([['日期', 'zuhe']]):
            if len(rqll) >= 4:
                inde = [i + 1 for i in range(4)]
                bjlist_xx = pd.DataFrame()
                for j in inde:
                    zuhelist = zuhel[j - 1]
                    zuhelist1 = re.findall(r'[01]', str(zuhelist))
                    bjlist_xx.at[0, 'rq_%d' % j] = rqll[j - 1]
                    bjlist_xx.at[0, 'zuhe_%d' % j] = ' '.join([str(x) for x in zuhelist1])
                bjlist_x1 = pd.concat([bjlist_x1, bjlist_xx])
                rqll.pop(0)
                zuhel.pop(0)
            rqll.append(row.日期)
            zuhel.append(row.zuhe)
            bjlist_x_shape = bjlist_x.shape
            if bjlist_x_shape[0] > 0 \
                    and row.日期 == bjlist_x.loc[bjlist_x_shape[0] - 1, '日期']:
                bjlist_xx1 = pd.DataFrame()
                for j in inde:
                    zuhelist = zuhel[j - 1]
                    zuhelist1 = re.findall(r'[01]', str(zuhelist))
                    bjlist_xx1.at[0, 'rq_%d' % j] = rqll[j - 1]
                    bjlist_xx1.at[0, 'zuhe_%d' % j] = ' '.join([str(x) for x in zuhelist1])
                bjlist_x1 = pd.concat([bjlist_x1, bjlist_xx1])
        bjlist_x1['rq'] = bjlist_x1['rq_4']
        bjlist_x1['zuhe'] = bjlist_x1.apply(lambda row2: row2.zuhe_1 + ' ' + row2.zuhe_2 + ' ' + row2.zuhe_3, axis=1)
        return bjlist_x1

    def get_gp_x_predict(self):
        bjlist_x = self.get_gp_akdata(xlorcs='predict')
        if bjlist_x.shape[0] == 0:
            return pd.DataFrame()
        encoder = self.binary_encoder(5)
        zhencoder = self.zuhe_encoder()
        bjlist_x['zhenfu'] = bjlist_x.apply(lambda row2: encoder(round(float(row2.振幅))), axis=1)
        bjlist_x['zhangdiefu'] = bjlist_x.apply(lambda row3: encoder(round(float(row3.涨跌幅) + 10)), axis=1)
        bjlist_x['huanshoulv'] = bjlist_x.apply(lambda row2: encoder(round(float(row2.换手率))), axis=1)
        bjlist_x['zuhe'] = bjlist_x.apply(lambda row2:
                                          zhencoder([encoder(round(float(row2.振幅))),
                                                     encoder(round(float(row2.涨跌幅) + 10)),
                                                     encoder(round(float(row2.换手率)))]), axis=1)
        rqll, zuhel = [], []
        bjlist_x1 = pd.DataFrame()
        for row in bjlist_x.itertuples([['日期', 'zuhe']]):
            inde = [i + 1 for i in range(3)]
            rqll.append(row.日期)
            zuhel.append(row.zuhe)
            bjlist_x_shape = bjlist_x.shape
            if bjlist_x_shape[0] > 0 \
                    and row.日期 == bjlist_x.loc[bjlist_x_shape[0] - 1, '日期']:
                bjlist_xx1 = pd.DataFrame()
                for j in inde:
                    zuhelist = zuhel[j - 1]
                    zuhelist1 = re.findall(r'[01]', str(zuhelist))
                    bjlist_xx1.at[0, 'rq_%d' % j] = rqll[j - 1]
                    bjlist_xx1.at[0, 'zuhe_%d' % j] = ' '.join([str(x) for x in zuhelist1])
                bjlist_x1 = pd.concat([bjlist_x1, bjlist_xx1])
        bjlist_x1['rq'] = bjlist_x1['rq_3']
        bjlist_x1['zuhe'] = bjlist_x1.apply(lambda row2: row2.zuhe_1 + ' ' + row2.zuhe_2 + ' ' + row2.zuhe_3, axis=1)
        return bjlist_x1

    def get_gp_xy(self):
        bjlist_x3 = pd.DataFrame()
        bjlist_x = self.load_gp_xlsdata()
        bjlist_x1 = self.get_gp_x()
        bjlist = self.get_gp_y()
        if bjlist_x.shape[0] == 0 or bjlist_x1.shape[0] == 0 or bjlist.shape[0] == 0:
            return pd.DataFrame()
        for row in bjlist_x.itertuples():
            bjlist_x3_1 = {'日期': [row.日期], '开盘': [row.开盘], '收盘': [row.收盘], '最高': [row.最高],
                           '最低': [row.最低], '成交量': [row.成交量], '成交额': [row.成交额], '振幅': [row.振幅],
                           '涨跌幅': [row.涨跌幅], '涨跌额': [row.涨跌额], '换手率': [row.换手率],
                           'zhenfu': [row.zhenfu], 'zhangdiefu': [row.zhangdiefu], 'huanshoulv': [row.huanshoulv],
                           'zuhe': [''], 'y': [-1]}
            for rowx in bjlist_x1.itertuples():
                if rowx.rq == row.日期:
                    bjlist_x3_1['zuhe'] = [rowx.zuhe]
            for rowy in bjlist.itertuples():
                if rowy.rq_1 == row.日期:
                    bjlist_x3_1['y'] = [rowy.y]
                    bjlist_x3_1['y1'] = [rowy.y1]
            bjlist_x3_2 = pd.DataFrame(bjlist_x3_1)
            bjlist_x3 = pd.concat([bjlist_x3, bjlist_x3_2])
        if bjlist_x3.shape[0] > 0:
            try:
                bjlist_x3.to_excel('./data/%s_xy.xlsx' % self.gpdmmc)
            except:
                print('保存%s_xy.xlsx失败' % self.gpdmmc)
        return bjlist_x3

    def getData(self):
        bjlist_x3 = pd.DataFrame()
        try:
            stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str(self.gpdm).zfill(6), period="daily",
                                                    start_date=self.startdate, end_date=self.enddate,
                                                    adjust="")
            encoder = self.binary_encoder(5)
            zhencoder = self.zuhe_encoder()
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
                                                         and row1['zdf_2'] > -1 and row1['zdf_3'] > -1 and row1[
                                                             'zdf_4'] > -1 else 0, axis=1)
            bjlist_x = stock_zh_a_hist_df
            bjlist_x['zhenfu'] = bjlist_x.apply(lambda row2: encoder(round(float(row2.振幅))), axis=1)
            bjlist_x['zhangdiefu'] = bjlist_x.apply(lambda row3: encoder(round(float(row3.涨跌幅) + 10)), axis=1)
            bjlist_x['huanshoulv'] = bjlist_x.apply(lambda row2: encoder(round(float(row2.换手率))), axis=1)
            bjlist_x['zuhe'] = bjlist_x.apply(lambda row2:
                                              zhencoder([encoder(round(float(row2.振幅))),
                                                         encoder(round(float(row2.涨跌幅) + 10)),
                                                         encoder(round(float(row2.换手率)))]), axis=1)
            rqll, zuhel = [], []
            bjlist_x1 = pd.DataFrame()
            for row in bjlist_x.itertuples([['日期', 'zuhe']]):
                if len(rqll) >= 4:
                    inde = [i + 1 for i in range(4)]
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
            bjlist_x1['rq'] = bjlist_x1['rq_4']
            bjlist_x1['zuhe'] = bjlist_x1.apply(lambda row2: row2.zuhe_1 + ' ' + row2.zuhe_2 + ' ' + row2.zuhe_3,
                                                axis=1)
            for row in bjlist_x.itertuples():
                bjlist_x3_1 = {'日期': [row.日期], '开盘': [row.开盘], '收盘': [row.收盘], '最高': [row.最高],
                               '最低': [row.最低], '成交量': [row.成交量], '成交额': [row.成交额], '振幅': [row.振幅],
                               '涨跌幅': [row.涨跌幅], '涨跌额': [row.涨跌额], '换手率': [row.换手率],
                               'zhenfu': [row.zhenfu],
                               'zhangdiefu': [row.zhangdiefu], 'huanshoulv': [row.huanshoulv], 'zuhe': ['']}
                for rowx in bjlist_x1.itertuples():
                    if rowx.rq == row.日期:
                        bjlist_x3_1['zuhe'] = [rowx.zuhe]
                bjlist_x3_1['y'] = ''
                for rowy in bjlist.itertuples():
                    if rowy.rq_1 == row.日期:
                        bjlist_x3_1['y'] = [rowy.y]
                bjlist_x3_2 = pd.DataFrame(bjlist_x3_1)
                bjlist_x3 = pd.concat([bjlist_x3, bjlist_x3_2])
            stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
        except ValueError:
            print('error', ValueError)
        return bjlist_x3

    def training_test_gen(self, x, y):
        assert len(x) == len(y)
        indices = np.random.permutation(range(len(x)))
        split_size = int(0.9 * len(indices))
        trX = x[indices[:split_size]]
        trY = y[indices[:split_size]]
        teX = x[indices[split_size:]]
        teY = y[indices[split_size:]]
        return trX, trY, teX, teY

    def get_pytorch_data(self):
        x = []
        y = []
        xylist = self.get_gp_xy()
        for row in xylist.itertuples():
            if len(row.zuhe) > 0 and row.y > -1:
                ret1 = [int(i) for i in str(row.zuhe).split(' ')]
                # ret2 = [int(i) for i in str(row.y).split(' ')]
                x.append(ret1)
                y.append(row.y)
        return self.training_test_gen(np.array(x), np.array(y))
