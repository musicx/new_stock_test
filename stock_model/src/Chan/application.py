import numpy as np
import pandas as pd
import QUANTAXIS as qtx
import datetime as dt
from Chan.data_structure import *

import matplotlib.pyplot as plt

def extract_data(code_list, start_date, end_date):
    candles = qtx.QA_fetch_stock_day_adv(code_list, start=start_date, end=end_date).to_qfq()
    charts = []
    #df = candles.data.loc[:, ['open', 'close', 'high', 'low']]
    try:
        candles.data['ds'] = [x.strftime('%Y-%m-%d') for x in candles.data.reset_index().date.tolist()]
    except:
        candles.data['ds'] = [x.strftime('%Y-%m-%d') for x in candles.data.date.tolist()]
    for idx, candle in enumerate(candles.splits()):
        chart = ChartFrame(code_list[idx], 'day')
        chart.init_candles(candle.data.loc[:, ['open', 'close', 'high', 'low']].values, candle.data.ds.tolist())
        charts.append(chart)
    return candles, charts



if __name__ == '__main__':
    end_date = dt.datetime.today()
    # end_date = dt.datetime(2020, 10, 23)
    start_date = end_date - dt.timedelta(days=365)

    # stocks = ['000528', '002049', '300529', '300607', '600518', '600588', '603877']
    # stocks = ['300638', '600516']
    stocks = ['603279']

    candles, charts = extract_data(stocks, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    boundaries = charts[0].detect_price_boundary()
    print(boundaries)





