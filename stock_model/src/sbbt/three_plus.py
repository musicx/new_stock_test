from __future__ import absolute_import, division, print_function, unicode_literals

import datetime  # For datetime objects
from quantax.data_query_advance import local_get_stock_day_adv
from sbbt.utilities import BasicTradeStats, prepare_stock, TradeList, prepare_stock_week
# Import the backtrader platform
import backtrader as bt

from tabulate import tabulate

# Create a Stratey
class ThreePlusStrategy(bt.Strategy):
    params = (
        ('short', 20),
        ('long', 60),
        ('exit', 5),
        ('gap', 0.03),
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

        # Add signal indicators
        self.short = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.short)
        self.long = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.long)
        self.exit = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.exit)
        self.sar = bt.indicators.ParabolicSAR(self.datas[0], af=0.02, afmax=0.1, plot=False)
        self.mtm = bt.indicators.MomentumOscillator(self.datas[0], period=20, plot=False)

        self.week_short = bt.indicators.SimpleMovingAverage(self.datas[1], period=self.p.short)
        self.week_long = bt.indicators.SimpleMovingAverage(self.datas[1], period=self.p.long)

        self.pass_exit = False

        # Indicators for the plotting show
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
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if (self.dataclose > self.short
                    and self.short > self.long
                    and self.dataclose > self.sar
                    and self.dataclose < self.short*(1+self.p.gap)
                    and self.short > self.short[-1]
                    and abs(self.mtm - 100) < 10
                    and self.week_short > self.week_long
                    and self.dataclose > self.week_short
            ):
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

                if self.dataclose > self.exit:
                    self.pass_exit = True
        else:
            if not self.pass_exit and self.dataclose > self.exit:
                self.pass_exit = True
            if (self.pass_exit and self.dataclose < self.exit) or (not self.pass_exit and self.dataclose < self.short):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
                self.pass_exit = False

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(ThreePlusStrategy)

    stock_code = '600143'
    # Create a Data FeedRatio
    stock_data = prepare_stock(stock_code)
    print('data samples:\n', stock_data.head())

    data = bt.feeds.PandasData(dataname=stock_data,
                               fromdate=datetime.datetime(2014, 1, 1),
                               todate=datetime.datetime(2020, 11, 30))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    stock_data_week = prepare_stock_week(stock_code)
    print('data resamples:\n', stock_data_week.head())
    week_data = bt.feeds.PandasData(dataname=stock_data_week,
                                    fromdate=datetime.datetime(2014, 1, 1),
                                    todate=datetime.datetime(2020, 11, 30))
    cerebro.adddata(week_data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)
    cerebro.broker.set_coc(True)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.PercentSizer, percents=99)

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
