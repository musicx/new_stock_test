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


def LLT_JCSC(data):
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
    try:
        data = QA.QA_fetch_stock_day_adv(stock, start='2010-01-01', end='2018-06-08').to_qfq()
        if data.data.shape[0] < 100:
            return -100, None, None
        # add indicator
        ind = data.add_func(CCI_JCSC)
        ind['ACT_JC'] = QA.REF(ind.JC, 1)
        ind['ACT_SC'] = QA.REF(ind.SC, 1)
        # print(ind.loc[(ind.ACT_JC > 0) | (ind.ACT_SC > 0), :])
    except:
        print('error for stock {}'.format(stock))
        return -100.0, None, None

    data_chosen = data.select_time('2010-03-01', '2018-06-08')

    has_hold = False
    points = []
    periods = []
    raise_percent = 1.
    last_date = ''
    for idx, item in enumerate(data_chosen.data.index):
        ochl = data_chosen.data.loc[item, ['open', 'close', 'high', 'low']]
        if ind.loc[item, 'ACT_JC'] > 0 and not has_hold and ochl.low != ochl.high:
            points.append((last_date, idx, ochl, 'buy', stock))
            has_hold = True
        if ind.loc[item, 'ACT_SC'] > 0 and has_hold and ind.loc[item, 'CZ'] != 1:
            periods.append((points[-1][0], str(item[0])[:10], idx - points[-1][1] + 1, ochl.close / points[-1][2].open - 1, stock))
            raise_percent *= ochl.close / points[-1][2].open
            if print_detail:
                print('{}: from {} to {}, change {}'.format(stock, periods[-1][0], periods[-1][1], periods[-1][3]))
            points.append((str(item[0])[:10], idx, ochl, 'sell', stock))
            has_hold = False
        last_date = str(item[0])[:10]
    if has_hold:
        periods.append((points[-1][0], str(item[0])[:10], idx - points[-1][1] + 1, ochl.close / points[-1][2].open - 1, stock))
        raise_percent *= ochl.close / points[-1][2].open
        if print_detail:
            print('{}: from {} to {}, change {}'.format(stock, periods[-1][0], periods[-1][1], periods[-1][3]))
        points.append((str(item[0])[:10], idx, ochl, 'sell', stock))
    print('{}: full range {}'.format(stock, raise_percent))
    return raise_percent, pd.DataFrame(periods, columns=['start', 'end', 'span', 'return', 'code'])


if __name__ == '__main__':
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
    print(history)
