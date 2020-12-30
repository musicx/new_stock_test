from __future__ import absolute_import, division, print_function, unicode_literals

import datetime  # For datetime objects
from quantax.data_query_advance import local_get_stock_day_adv, local_get_index_day_adv

import numpy as np
import pandas as pd

# Import the backtrader platform
import backtrader as bt
from sbbt.sar import SAR


def prepare_data(code):
    print('reading data... {}'.format(code))
    data = local_get_stock_day_adv(code, start=datetime.date(2010, 1, 1).strftime('%Y-%m-%d'),
                                   end=datetime.datetime.today().strftime('%Y-%m-%d')).to_qfq()
    formatted = data.data.reset_index().loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']]
    formatted.rename(columns={'date': 'datetime'}, inplace=True)
    return formatted.set_index('datetime')

# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        self.sar = SAR(af=0.02, afmax=0.2)

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f; Sar, %.3f' % (self.dataclose[0], self.sar[0]))

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    stock_data = prepare_data('601985')
    print('data samples:\n', stock_data.head())

    data = bt.feeds.PandasData(dataname=stock_data,
                               fromdate=datetime.datetime(2019, 1, 1),
                               todate=datetime.datetime(2020, 12, 30))
    cerebro.adddata(data)

    cerebro.run()
