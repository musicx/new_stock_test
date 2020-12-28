from __future__ import absolute_import, division, print_function, unicode_literals

import datetime  # For datetime objects
from quantax.data_query_advance import local_get_stock_day_adv

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
class BiasRatioCrossStrategy(bt.Strategy):
    params = (
        ('short', 20),
        ('mid', 60),
        ('long', 120)
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
        self.short = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.short)
        self.mid = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.mid)
        self.long = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.long)

        self.csr = (self.dataclose / self.short) - 1
        self.smr = (self.short / self.mid) - 1
        self.mlr = (self.mid / self.long) - 1

        # Indicators for the plotting show
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.ATR(self.datas[0], plot=False)

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
            if self.csr > self.smr and self.smr > self.mlr and self.smr > self.smr[-1] and self.csr[-1] < self.smr[-1]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.csr < self.smr or self.csr < self.mlr or self.csr < 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()



def prepare_data(code):
    print('reading data... {}'.format(code))
    data = local_get_stock_day_adv(code, start=datetime.date(2010, 1, 1).strftime('%Y-%m-%d'),
                                   end=datetime.datetime.today().strftime('%Y-%m-%d')).to_qfq()
    formatted = data.data.reset_index().loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']]
    formatted.rename(columns={'date': 'datetime'}, inplace=True)
    return formatted.set_index('datetime')


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(BiasRatioCrossStrategy)

    # Create a Data FeedRatio
    stock_data = prepare_data('000723')
    print('data samples:\n', stock_data.head())

    data = bt.feeds.PandasData(dataname=stock_data,
                               fromdate=datetime.datetime(2014, 1, 1),
                               todate=datetime.datetime(2020, 11, 30))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.PercentSizer, percents=80)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()