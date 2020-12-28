# coding:utf-8

from quantax.data_query_advance import local_get_stock_day_adv, local_get_stock_list_adv, local_get_stock_min_adv
import pandas as pd
import numpy as np
import datetime as dt
from Chan.application import extract_data_1d
from Chan.indicator import ChanIndicator

if __name__ == '__main__':
    end_date = dt.datetime.today()
    # end_date = dt.datetime(2020, 10, 23)
    start_date = end_date - dt.timedelta(days=365)
    stocks = ['000528', '002049', '300529', '300607', '600518', '600588', '603877']

    candles, charts = extract_data_1d(stocks, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    indicator = ChanIndicator()
    near_lines = [(chart.code, indicator.near_pivot_lines(chart)) for chart in charts]
    chosen = [(code, line) for code, line in near_lines if line > 0]
    print(chosen)