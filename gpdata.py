import akshare as ak
import pandas as pd


def biaozhunhua(a):
    if a < -10:
        return -3
    if -10 <= a < -5:
        return -2
    if -5 <= a < 0:
        return -1
    if 0 <= a < 5:
        return 0
    if 5 <= a < 10:
        return 1
    if 10 <= a < 20:
        return 3
    if a > 20:
        return 4


stock_zh_index_daily_df = pd.DataFrame()
try:
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str('2529').zfill(6), period="daily",
                                            start_date="20230101", end_date='20231231',
                                            adjust="")
    xuhao = 0
    dailydf = pd.DataFrame()
    for row in stock_zh_a_hist_df.itertuples():
        dailydfx = {'Index': row.Index, '日期': row.日期, '收盘': row.收盘,
                    '振幅': float(row.振幅),
                    '涨跌幅': float(row.涨跌幅),
                    '换手率': float(row.换手率),
                    'zhenfu': biaozhunhua(float(row.振幅)),
                    'zhangdiefu': biaozhunhua(float(row.涨跌幅)),
                    'huanshoulv': biaozhunhua(float(row.换手率))}
        dailydfx1 = pd.DataFrame(dailydfx, index=[0])
        dailydf = pd.concat([dailydf, dailydfx1])
    dailydf.to_excel('ceshi.xlsx')
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
    print(1)
except ValueError:
    print('error', ValueError)
