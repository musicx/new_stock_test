import numpy as np
import pandas as pd
import QUANTAXIS as QA
from joblib import Parallel, delayed
from pprint import pprint as print


def llt(price, alpha):
    low_latency = []
    for idx, pr in enumerate(price):
        if idx == 0:
            low_latency.append(pr)
        elif idx == 1:
            low_latency.append(alpha*pr + (1-alpha)*price[0])
        else:
            low_latency.append((alpha-alpha**2/4)*pr+(alpha**2/2)*price[idx-1]-(alpha-3*(alpha**2)/4)*price[idx-2]
                               + 2*(1-alpha)*low_latency[idx-1]-(1-alpha)**2*low_latency[idx-2])
    return np.array(low_latency)


def JCSC(data):
    """
    1.ema3 上穿 llt0.05 或 llt0.1＞llt0.05时 ema3 上穿 llt0.1  买入
    2.ema3 下穿 llt0.1 或 llt0.1 向下  卖出
    """
    data['llt1'] = llt(data.loc[:, 'close'].values, 0.05)
    data['llt2'] = llt(data.loc[:, 'close'].values, 0.1)
    llt2_ref = QA.REF(data.llt2, 1)

    ema5 = QA.EMA(data.loc[:, 'close'], 5)
    #ma5 = QA.MA(data.loc[:, 'close'], 5)

    up1 = QA.CROSS(ema5, data.llt1)
    up2 = QA.CROSS(ema5, data.llt2)
    down1 = QA.CROSS(data.llt1, ema5)
    down2 = QA.CROSS(llt2_ref, data.llt2)

    jc1 = pd.concat([up2, data.llt2 > data.llt1], axis=1).all(axis=1)
    JC = pd.concat([up1, jc1], axis=1).any(axis=1) * 1
    SC = pd.concat([down1, down2], axis=1).any(axis=1) * 1
    return pd.DataFrame({'LLT1': data.llt1, 'LLT2': data.llt2, 'EMA3': ema5, 'JC': JC, 'SC': SC})


def CCI_JCSC(data):
    """
    1.cci 上穿 -100 且 MACD DIFF向上  买入
    2.cci 下穿 -100， 0， 100 卖出
    """
    cci = QA.QA_indicator_CCI(data, N=14)
    cci['low'] = -100
    cci['mid'] = 0
    cci['up'] = 100
    macd = QA.QA_indicator_MACD(data, short=12, long=26, mid=9)

    up1 = QA.CROSS(cci.CCI, cci.low)
    # up2 = macd.DIF > macd.DIF.shift(1)
    up2 = macd.DIF == macd.DIF
    up3 = QA.CROSS(cci.CCI, cci.mid)
    down1 = QA.CROSS(cci.low, cci.CCI)
    down2 = QA.CROSS(cci.mid, cci.CCI)
    down3 = QA.CROSS(cci.up, cci.CCI)

    CZ = pd.concat([up3, up2], axis=1).all(axis=1) * 1
    JC = pd.concat([up1, up2], axis=1).all(axis=1) * 1
    SC = pd.concat([down1, down2, down3], axis=1).any(axis=1) * 1
    return pd.DataFrame({'CCI': cci.CCI, 'DIFF': macd.DIF, 'JC': JC, 'SC': SC, 'CZ': CZ})



def backtest(stock, print_detail):
    Account = QA.QA_Account()
    Broker = QA.QA_BacktestBroker()

    Account.reset_assets(100000)
    Account.account_cookie = 'user_admin_llt'
    try:
        data = QA.QA_fetch_stock_day_adv(stock, start='2010-01-01', end='2018-06-06').to_qfq()
        if data.data.shape[0] < 100:
            return -100, None
        # add indicator
        ind = data.add_func(CCI_JCSC)
        ind['ACT_JC'] = QA.REF(ind.JC, 1)
        ind['ACT_SC'] = QA.REF(ind.SC, 1)
        # print(ind.loc[(ind.ACT_JC > 0) | (ind.ACT_SC > 0), :])
        # ind.xs('000001',level=1)['2018-01'].plot()

        data_forbacktest = data.select_time('2010-03-01', '2018-06-06')
        last_close = 0
        for items in data_forbacktest.panel_gen:
            for item in items.security_gen:
                daily_ind = ind.loc[item.index]
                if str(item.data.date[0])[:10] == '2018-01-26' and print_detail:
                     print('inspect day error!')
                if daily_ind.ACT_JC.iloc[0] > 0 and Account.sell_available.get(stock, 0) == 0 and item.data.high[0] != item.data.low[0]:
                    if print_detail:
                        print('{} buy at {}'.format(item.data.date[0], item.data.open[0]))
                    possible = int(item.data.open[0] * 10000) / 10000.
                    if possible < item.data.low[0]:
                        possible = np.ceil(possible * 10000 + 1) / 10000.
                    elif possible > item.data.high[0]:
                        possible = np.floor(possible * 10000 - 1) / 10000.
                    order = Account.send_order(
                        code=item.data.code[0],
                        time=item.data.date[0],
                        amount=Account.cash_available / (np.ceil(possible * 100) / 100.) / 1.0003 // 100 * 100,
                        towards=QA.ORDER_DIRECTION.BUY,
                        price=possible,
                        order_model=QA.ORDER_MODEL.LIMIT,
                        amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
                    )
                    Account.receive_deal(Broker.receive_order(QA.QA_Event(order=order, market_data=item)))
                elif daily_ind.ACT_SC.iloc[0] > 0 and daily_ind.CZ.iloc[0] != 1:
                    if Account.sell_available.get(stock, 0) > 0:
                        if print_detail:
                            print('{} sell at {}'.format(item.data.date[0], item.data.close[0]))
                        order = Account.send_order(
                            code=item.data.code[0],
                            time=item.data.date[0],
                            amount=Account.sell_available.get(stock, 0),
                            towards=QA.ORDER_DIRECTION.SELL,
                            price=0,
                            order_model=QA.ORDER_MODEL.CLOSE,
                            amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
                        )
                        Account.receive_deal(Broker.receive_order(QA.QA_Event(order=order, market_data=item)))
                last_close = item.data.close[0]
            Account.settle()
    except:
        print('error for stock {}'.format(stock))
        return -100.0, None

    end_cash = Account.latest_cash + Account.sell_available.get(stock, 0) * last_close
    end_return = end_cash / 100000 - 1
    if end_return == 0:
        print('nothing happened at {}'.format(stock))
        return -100.0, None
    # returns.append(end_return)
    Risk = QA.QA_Risk(Account)
    if print_detail:
        # print('account history')
        # print(Account.history)
        print('account history table')
        print(Account.history_table)
        # print('account daily hold')
        # print(Account.daily_hold)

        # create Risk analysis
        # Risk = QA.QA_Risk(Account)
        print('risk message')
        print(Risk.message)
        print(Risk.message['max_dropback'])
        print('risk assets')
        print(Risk.assets)

        print('asset at the end:')
        print(end_cash)

        Risk.plot_assets_curve()
        # Risk.plot_dailyhold()
        # Risk.plot_signal()
        # Risk.assets.plot()
        # Risk.benchmark_assets.plot()


        # account_info = QA.QA_fetch_account({'account_cookie': 'user_admin_macd'})
        # account = QA.QA_Account().from_message(account_info[0])
        # print(account)
        #
        # # save result
        # Account.save()
        # Risk.save()
    else:
        # Risk = QA.QA_Risk(Account)
        print('Stock {}: end {}, drop {}'.format(stock, end_cash, Risk.message['max_dropback']))
    return end_return, Account.history_table


if __name__ == '__main__':
    # create account
    PRINT_DETAIL = False
    # get data from mongodb
    stocks = QA.QA_fetch_stock_list_adv().code.tolist()
    # stocks = ['002762', '002458', '002133', '600115']
    # stocks = ['002458']
    returns = Parallel(n_jobs=4)(delayed(backtest)(stock, False) for stock in stocks)
    # returns = [backtest(stock, True) for stock in stocks]
    rets = np.array([r[0] for r in returns if r[0] != -100])
    history = pd.concat([r[1] for r in returns if r[0] != -100])
    history.to_hdf('../data/cci_long.hdf', 'data')
    print('\navg returns: {}, std: {}'.format(rets.mean(), rets.std()))
    rets.sort()
    print('\nmid returns: {}'.format(rets[len(rets) // 2]))
    # print(history)
