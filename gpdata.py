import akshare as ak
import pandas as pd

stock_zh_index_daily_df = pd.DataFrame()
try:
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str('2529').zfill(6), period="daily",
                                            start_date="20230101", end_date='20231231',
                                            adjust="")
    stock_zh_a_hist_df.to_excel('ceshi.xlsx')
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
    print(1)
except:
    print('error')