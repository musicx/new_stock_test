import numpy as np
import pandas as pd
import tensorflow as tf
import QUANTAXIS as qa

periods = 30
pvars = ['open', 'close', 'high', 'low']
vvars = ['vol']

stocks = qa.QA_fetch_stock_list_adv()
train = []
test = []
for stock in stocks.code.tolist():
    try:
        candles = qa.QA_fetch_stock_day_adv(stock, start='2014-01-01', end='2018-05-02')
        cdata = candles.data.reset_index(level=1, drop=True).loc[:, pvars]
        data = pd.concat([cdata - cdata.diff(x) for x in range(periods)], axis=1).apply(lambda x: x / cdata.close) - 1
        cdata2 = candles.data.reset_index(level=1, drop=True).loc[:, vvars]
        data2 = pd.concat([cdata2 - cdata2.diff(x) for x in range(periods)], axis=1).apply(lambda x: x / cdata2.vol)
        change = (cdata - cdata.diff(-3)).close / (cdata - cdata.diff(-1)).open - 1
        # label = pd.concat([(change > 0.1), (change <= 0.1) & (change > 0),
        #                    (change <= 0) & (change > -0.1), (change <= -0.1)], axis=1) * 1
        label = (change > 0.1) * 1 + (change > 0) * 1 + (change > -0.1) * 1

        all_data = pd.concat([data, data2, label], axis=1).dropna()
        print('stock {} have {} examples, left {} examples'.format(stock, data.shape[0], all_data.shape[0]))
        if all_data.shape[0] == 0:
            continue
        all_data['code'] = stock

        tr = all_data.truncate(after='2018-01-01')
        ts = all_data.truncate(before='2018-01-01')
        if tr.shape[0] > 0:
            train.append(tr)
        if ts.shape[0] > 0:
            test.append(ts)
    except:
        pass

train = pd.concat(train, axis=0).reset_index()
test = pd.concat(test, axis=0).reset_index()

train.to_csv('../data/train_14_17.csv', index=False)
test.to_csv('../data/test_14_17.csv', index=False)
