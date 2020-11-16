import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from functools import reduce


def go_up(data, N=5, P=3, M=0.1):
    ind = (data.close > data.close.shift(1)) * 1
    checked = check(ind, -N)
    lift = data.high.rolling(N).max().shift(-N-1) / data.open.shift(-1)
    flat = ((data.close == data.open) & (data.close == data.high)) * 1
    return pd.DataFrame({'go_up': (~(flat > 0) & ~(flat.shift(-1) > 0) & (checked >= P) & (lift > (1+M))) * 1})


def check(signal, days):
    return reduce(lambda x, y: x + y, [signal.shift(x).fillna(0)
                                       for x in (range(days) if days >= 0 else range(-1, days-1, -1))])


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    today_str = today.strftime('%Y-%m-%d')

    # divide_date = today - dt.timedelta(days=60)
    # divide_str = divide_date.strftime('%Y-%m-%d')

    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()
    for stock in stock_list[:10]:
        print('handling %s' % stock)
        try:
            candles = qa.QA_fetch_stock_day_adv(stock, start='2013-01-01', end=today_str).to_qfq()
            if candles.data.shape[0] <= 100:
                continue
        except:
            print('data error during {}'.format(stock))
            continue

        data = candles.data
        data['label'] = candles.add_func(go_up)

        if data.shape[0] > 200:
            train = data.iloc[100:-100, :]
            test = data.iloc[-100:, :]
        elif data.shape[0] > 100:
            train = None
            test = data.iloc[100:, :]
        else:
            train = None
            test = None

