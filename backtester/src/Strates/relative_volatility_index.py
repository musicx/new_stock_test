# coding:utf-8
import math
import numpy as np
import pandas as pd
import datetime  # For datetime objects
from quantax.data_query_advance import local_get_stock_day_adv
from Strates.utility import BasicTradeStats, prepare_stock, TradeList
import backtrader as bt
from tabulate import tabulate

class RVI(bt.Indicator):
    lines = ('rvi',)
    params = (
        ('std_period', 7),
        ('ema_period', 7),
    )

    def __init__(self):
        std = bt.ind.StandardDeviation(self.data, period=self.p.std_period)
        pos = bt.If(self.data.close > self.data.close(-1), std, 0)
        neg = bt.If(self.data.close < self.data.close(-1), std, 0)
        usum = bt.ind.SumN(pos, period=self.p.std_period)
        dsum = bt.ind.SumN(neg, period=self.p.std_period)
        uavg = bt.ind.ExponentialMovingAverage(usum, period=self.p.ema_period)
        davg = bt.ind.ExponentialMovingAverage(dsum, period=self.p.ema_period)
        self.l.rvi = bt.ind.DivByZero(uavg, (uavg + davg)) * 100


class RVICrossStrategy(bt.Strategy):
    params = (
        ('std_period', 7),
        ('ema_period', 13),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add MovingAverageSimple indicator
        self.ma20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)
        self.ma60 = bt.indicators.SimpleMovingAverage(self.datas[0], period=60)
        self.rvi = RVI(self.data, std_period=self.p.std_period, ema_period=self.p.ema_period, subplot=True)

        # Indicators for the plotting show
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        #self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        up = 50
        down = 50

        # Check if we are in the market
        if not self.position:
            if self.rvi.rvi[0] > up and self.rvi.rvi[-1] < up and self.rvi.rvi[-2] < up:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
        else:
            if self.rvi.rvi[0] < down and self.rvi.rvi[-1] > down:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(RVICrossStrategy)

    # Create a Data FeedRatio
    stock_data = prepare_stock('000758')
    print('data samples:\n', stock_data.head())

    data = bt.feeds.PandasData(dataname=stock_data,
                               fromdate=datetime.datetime(2014, 1, 1),
                               todate=datetime.datetime(2020, 12, 30))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.PercentSizer, percents=90)

    cerebro.addanalyzer(BasicTradeStats, _name='basic')
    cerebro.addanalyzer(TradeList, _name='trade')

    # Set the commission
    cerebro.broker.setcommission(commission=0.0001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    results = cerebro.run(tradehistory=True)

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    basic = results[0].analyzers.getbyname('basic')
    basic.print()

    trade = results[0].analyzers.getbyname('trade').get_analysis()
    print(tabulate(trade, headers="keys"))

    # Plot the result
    #cerebro.plot()
