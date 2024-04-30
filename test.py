import pandas as pd
import akshare as ak
import datetime
import numpy as np
import re
import dangegupiao.gpfileutil as f

try:
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol='000001', period="daily",
                                            start_date='20240101', end_date='20240401',
                                            adjust="")
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol='000001')
    # stock_zh_a_hist_df.to_excel('./data/%s.xlsx' % self.gpdmmc)
except ValueError:
    print('error', ValueError)
print('1')
