import numpy as np
import pandas as pd
import QUANTAXIS as QA
from pprint import pprint as print


# define the MACD strategy
def MACD_JCSC(dataframe, SHORT=12, LONG=26, M=9):
    """
    1.DIF向上突破DEA，买入信号参考。
    2.DIF向下跌破DEA，卖出信号参考。
    """
    CLOSE = dataframe.close
    DIFF = QA.EMA(CLOSE, SHORT) - QA.EMA(CLOSE, LONG)
    DEA = QA.EMA(DIFF, M)
    MACD = 2*(DIFF-DEA)

    CROSS_JC = QA.CROSS(DIFF, DEA)
    CROSS_SC = QA.CROSS(DEA, DIFF)
    ZERO = 0
    return pd.DataFrame({'DIFF': DIFF, 'DEA': DEA, 'MACD': MACD, 'CROSS_JC': CROSS_JC, 'CROSS_SC': CROSS_SC, 'ZERO': ZERO})


# create account
Account = QA.QA_Account()
Broker = QA.QA_BacktestBroker()

Account.reset_assets(1000000)
Account.account_cookie = 'user_admin_macd'

# get data from mongodb
data = QA.QA_fetch_stock_day_adv(['000001', '000002', '000004', '600000'], '2017-09-01', '2018-05-20')
data = data.to_qfq()

# add indicator
ind = data.add_func(MACD_JCSC)
# ind.xs('000001',level=1)['2018-01'].plot()

data_forbacktest = data.select_time('2018-01-01', '2018-05-20')

for items in data_forbacktest.panel_gen:
    for item in items.security_gen:
        daily_ind = ind.loc[item.index]
        if daily_ind.CROSS_JC.iloc[0] > 0:
            order = Account.send_order(
                code=item.data.code[0],
                time=item.data.date[0],
                amount=1000,
                towards=QA.ORDER_DIRECTION.BUY,
                price=0,
                order_model=QA.ORDER_MODEL.CLOSE,
                amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
            )
            Account.receive_deal(Broker.receive_order(
                QA.QA_Event(order=order, market_data=item)))
        elif daily_ind.CROSS_SC.iloc[0] > 0:
            if Account.sell_available.get(item.code[0], 0) > 0:
                order = Account.send_order(
                    code=item.data.code[0],
                    time=item.data.date[0],
                    amount=Account.sell_available.get(item.code[0], 0),
                    towards=QA.ORDER_DIRECTION.SELL,
                    price=0,
                    order_model=QA.ORDER_MODEL.MARKET,
                    amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
                )
                Account.receive_deal(Broker.receive_order(
                    QA.QA_Event(order=order, market_data=item)))
    Account.settle()

print('account history')
print(Account.history)
print('account history table')
print(Account.history_table)
print('account daily hold')
print(Account.daily_hold)

# create Risk analysis
Risk = QA.QA_Risk(Account)
print('risk message')
print(Risk.message)
print('risk assets')
print(Risk.assets)
Risk.plot_assets_curve()
Risk.plot_dailyhold()
Risk.plot_signal()
# Risk.assets.plot()
# Risk.benchmark_assets.plot()

# save result
Account.save()
Risk.save()

account_info = QA.QA_fetch_account({'account_cookie': 'user_admin_macd'})
account = QA.QA_Account().from_message(account_info[0])
print(account)
