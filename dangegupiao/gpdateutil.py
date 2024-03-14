import akshare as ak
import datetime


def getstart_end(gpdm):
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol=gpdm)
    a = stock_individual_info_em_df.iloc[7, 1]
    now_time = datetime.datetime.now()
    yesterday_time = now_time - datetime.timedelta(days=1)
    stryesterday = yesterday_time.strftime('%Y%m%d')
    return [str(a), stryesterday]