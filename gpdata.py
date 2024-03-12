import akshare as ak
import pandas as pd

stock_zh_index_daily_df = pd.DataFrame()
try:
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str('2529').zfill(6), period="daily",
                                            start_date="20230101", end_date='20231231',
                                            adjust="")
    xuhao = 0
    dailydf = pd.DataFrame()
    for row in stock_zh_a_hist_df.itertuples():
        dailydfx = {'Index': row.Index, 'riqi': row.日期, 'shoupan': row.收盘, 'zhenfu': round(float(row.振幅)),
                    'zhangdiefu': round(float(row.涨跌幅)),
                    'huanshoulv': round(float(row.换手率))}
        dailydfx1 = pd.DataFrame(dailydfx, index=[0])
        dailydf = pd.concat([dailydf, dailydfx1])
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
    print(1)
except ValueError:
    print('error', ValueError)
