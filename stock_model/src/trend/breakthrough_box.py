import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from utility import *


if __name__ == '__main__':
    today = dt.datetime.today()
    start_date = today - dt.timedelta(days=365)

    # stocks = ['000528', '002049', '300529', '300607', '600518', '600588', '603877']
    # stocks = ['300638', '600516']
    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()

    with open('../../data/break_through_box.txt', 'w') as f:
        for stock in stock_list:
            try:
                can = qa.QA_fetch_stock_day_adv(stock, start=start_date.strftime('%Y-%m-%d'),
                                                end=today.strftime('%Y-%m-%d')).to_qfq()
            except AttributeError as e:
                continue
            raw = can.data.loc[:, ['open', 'close', 'high', 'low']]
            try:
                raw['dd'] = [x.strftime('%Y-%m-%d') for x in can.data.reset_index().date.tolist()]
            except:
                raw['dd'] = [x.strftime('%Y-%m-%d') for x in can.data.date.tolist()]
            klines = [Kline(x[0], x[1], x[2], x[3], x[4]) for x in raw.values]

            f.write('stock code: {}\n'.format(stock))

            merged = merge_klines(klines)
            ends = find_endpoints(merged)

            tops = [point[0] for point in ends if point[1] and point[2]]
            lines = []
            for idx, top in enumerate(tops):
                same = []
                for inner in tops[idx+1: idx+6]:
                    if merged[inner].high / merged[top].high > 1.06:
                        same.clear()
                        break
                    if abs(merged[inner].high / merged[top].high - 1) < 0.03:
                        same.append(inner)
                if len(same) > 0:
                    f.write('base: {} @ {}, found: {}\n'.format(merged[top].date, merged[top].high,
                                                                [(merged[x].date, merged[x].high) for x in same]))

                    line = min([merged[top].high] + [merged[x].high for x in same])
                    if raw.ix[-1, 'high'] >= line > raw.ix[-2, 'high']:
                        print("break through found!!! {}".format(stock))
                        f.write("break through found!!! {}\n".format(stock))

