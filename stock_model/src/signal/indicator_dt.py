import numpy as np
import pandas as pd
import QUANTAXIS as qa
from sklearn import linear_model
from functools import reduce
from pymodline.woe.bin_merger import BinMerger
from pymodline.woe.bin_splitter import BinSplitter

def skdj_buy(data):
    ind = pd.DataFrame({'SKDJ_BUY': pd.concat([data['SKDJ_K'] > data['SKDJ_D'],
                                               qa.REF(data['SKDJ_K'], 1) < qa.REF(data['SKDJ_D'], 1),
                                               qa.REF(data['SKDJ_D'], 1) < 20], axis=1).all(axis=1) * 1})
    return ind


def cci_buy(data, T=-100):
    ind = pd.DataFrame({'CCI_BUY': pd.concat([data['CCI'] > T,
                                              qa.REF(data['CCI'], 1) < T], axis=1).all(axis=1) * 1})
    return ind


def macd_buy(data):
    ind = pd.DataFrame({'MACD_BUY': pd.concat([data['DIF'] > data['DEA'],
                                               qa.REF(data['DIF'], 1) < qa.REF(data['DEA'], 1)], axis=1).all(axis=1) * 1})
    return ind


def ROC(data, N=25, M=6, E=9):
    avs = data['amount'] / data['volume']
    roc = 100 * (avs - qa.REF(avs, N)) / qa.REF(avs, N)
    maroc = qa.MA(roc, M)
    emaroc = qa.EMA(roc, E)
    return pd.DataFrame({'ROC': roc, 'MAROC': maroc, 'EMAROC': emaroc})


def roc_buy(data, D=5):

    cross = pd.concat([data['ROC'] > data['EMAROC'],
                       data['ROC'].shift(1) < data['EMAROC'].shift(1),
                       data['ROC'] < 0], axis=1).all(axis=1) * 1
    sum = reduce(lambda x, y: x + y, [cross.shift(x).fillna(0) for x in range(D)])
    ind = pd.DataFrame({'ROC_BUY': pd.concat([sum > 0,
                                              data['EMAROC'] > data['EMAROC'].shift(1),
                                              data['MAROC'] > data['MAROC'].shift(1),
                                              data['ROC'] > data['MAROC']], axis=1).all(axis=1) * 1})
    return ind


def LWR(data, N=9, M=3):
    rsv = (qa.HHV(data.high, N) - data.close) / (qa.HHV(data.high, N) - qa.LLV(data.low, N)) * 100
    slow = qa.SMA(rsv, M, 1)
    fast = qa.SMA(slow, M, 1)
    return pd.DataFrame({'FAST': fast, 'SLOW': slow})


def DTGF(data, N=21):
    fast = (data['low'] * 4 + data['close'] * 3 + data['open'] * 2 + data['high']) / 10.
    slow = qa.EMA(fast, N)
    return pd.DataFrame({'FAST': fast, 'SLOW': slow})


def BDZW(data, A=13, B=17, C=20):
    fast = qa.MA(data.close, A) + qa.MA(data.close, B) - qa.REF(qa.MA(data.close, C), 1)
    slow = fast * 2 - qa.EMA(data.close, 3)
    return pd.DataFrame({'FAST': fast, 'SLOW': slow})


def JMDD(data, N=36):
    Var1 = (data.close - qa.LLV(data.low, N)) / (qa.HHV(data.high, N) - qa.LLV(data.low, N)) * 100
    Var2 = qa.SMA(Var1, 3, 1)
    Var3 = qa.SMA(Var2, 3, 1)
    Var4 = qa.SMA(Var3, 3, 1)
    return pd.DataFrame({'FAST': Var3, 'SLOW': Var4})


def MDZ(data):
    qjj = data.volume / (((data.high - data.low) * 2) - qa.ABS(data.close - data.open))
    xvl = qa.IF(data.close > data.open, qjj * (data.high - data.low),
               qa.IF(data.close < data.open, qjj * (data.high - data.open + data.close - data.low), (data.volume / 2))) + \
          qa.IF(data.close > data.open, - qjj * (data.high - data.close + data.open - data.low),
                qa.IF(data.close < data.open, - (qjj * (data.high - data.low)), (0 - (data.volume / 2))))
    typm = xvl / 115.0
    trend = qa.EMA(data.close, 90)
    tred = qa.MA(typm, 15)
    return pd.DataFrame({'FAST': trend, 'SLOW': tred})


def normal_buy(data):
    ind = pd.DataFrame({'NORM_BUY': pd.concat([data['FAST'] > data['SLOW'],
                                               qa.REF(data['FAST'], 1) < qa.REF(data['SLOW'], 1)], axis=1).all(axis=1) * 1})
    return ind


def cross_boll_mid(data, boll):
    ind = pd.concat([data.high > boll['BOLL'], data.shift(1).high < boll['BOLL'].shift(1)], axis=1).all(axis=1) * 1
    return pd.DataFrame({'CROSS': ind})


def cross_boll_upper(data, boll):
    ind = pd.concat([data.high > boll['UB'], data.shift(1).high < boll['UB'].shift(1)], axis=1).all(axis=1) * 1
    return pd.DataFrame({'CROSS': ind})


def check(signal, days):
    return reduce(lambda x, y: x + y, [signal.shift(x).fillna(0)
                                       for x in (range(days) if days >= 0 else range(-1, days-1, -1))])



pfive = []
plabel = []
start = []
stocks = qa.QA_fetch_stock_list_adv()
stock_list = stocks.code.tolist()
for stock in stock_list[:1]:
    print('handling %s' % stock)
    try:
        candles = qa.QA_fetch_stock_day_adv(stock, start='2014-01-01', end='2018-05-22').to_qfq()
    except:
        print('data error during {}'.format(stock))
        continue

    # candles = qa.QA_fetch_stock_day_adv(stock, start='2014-01-01', end='2018-05-02').to_qfq()

    try:
        skdj_9_3 = candles.add_func(qa.QA_indicator_SKDJ, N=9, M=3)  # [['SKDJ_K', 'SKDJ_D']]  # .rename(columns={'SKDJ_D': 'SKDJ_D_9_3', 'SKDJ_K': 'SKDJ_K_9_3'})
        cci_14 = candles.add_func(qa.QA_indicator_CCI, N=14)  # ['CCI']    # .rename(columns={'CCI': 'CCI_14'})
        #cci_83 = candles.add_func(qa.QA_indicator_CCI, N=83)  # ['CCI']    # .rename(columns={'CCI': 'CCI_83'})
        macd_12_26 = candles.add_func(qa.QA_indicator_MACD, short=12, long=26, mid=9)  # [['DIF', 'DEA', 'MACD']]    # .rename(columns={'DIF': 'DIF_12_26', 'DEA': 'DEA_12_26', 'MACD': 'MACD_12_26'})
        boll_99 = candles.add_func(qa.QA_indicator_BOLL, N=99)
        roc_25 = candles.add_func(ROC, N=25)
        dtgf_21 = candles.add_func(DTGF, N=21)
        bdzw = candles.add_func(BDZW)
        jmdd = candles.add_func(JMDD)
        mdz = candles.add_func(MDZ)
        lwr_9 = candles.add_func(LWR, N=9, M=3)

        buys = [skdj_buy(skdj_9_3),
                cci_buy(cci_14).rename(columns={'CCI_BUY': 'CCI_BUY_14_low'}),
                cci_buy(cci_14, 0).rename(columns={'CCI_BUY': 'CCI_BUY_14_mid'}),
                # cci_buy(cci_14, 100).rename(columns={'CCI_BUY': 'CCI_BUY_14_100'}),
                # cci_buy(cci_83).rename(columns={'CCI_BUY': 'CCI_BUY_83_m100'}),
                # cci_buy(cci_83, 0).rename(columns={'CCI_BUY': 'CCI_BUY_83_0'}),
                # cci_buy(cci_83, 100).rename(columns={'CCI_BUY': 'CCI_BUY_83_100'}),
                roc_buy(roc_25),
                normal_buy(dtgf_21).rename(columns={'NORM_BUY': 'DTGF_BUY'}),
                normal_buy(bdzw).rename(columns={'NORM_BUY': 'BDZW_BUY'}),
                normal_buy(jmdd).rename(columns={'NORM_BUY': 'JMDD_BUY'}),
                normal_buy(mdz).rename(columns={'NORM_BUY': 'MDZ_BUY'}),
                normal_buy(lwr_9).rename(columns={'NORM_BUY': 'LWR_BUY'}),
                macd_buy(macd_12_26)]
        cross = candles.add_func(cross_boll_upper, boll=boll_99)
        cross['up'] = (check(cross, -20) > 0) * 1
        cross['mid'] = (check(candles.add_func(cross_boll_mid, boll=boll_99), -20) > 0) * 1
    except:
        print('func error during: {}'.format(stock))
        continue

    indicators = pd.concat(buys, axis=1)
    observe = pd.concat([indicators, indicators.shift(1).rename(columns=dict([(x, x + "_r1") for x in indicators.columns]))], axis=1)
    start.append(observe.loc[observe.sum(axis=1) > 5, :])
    # pfive.append(pd.concat([indicators,
    #                         indicators.shift(1).rename(columns=dict([(x, x + "_r1") for x in indicators.columns])),
    #                         indicators.shift(2).rename(columns=dict([(x, x + "_r2") for x in indicators.columns])),
    #                         indicators.shift(3).rename(columns=dict([(x, x + "_r3") for x in indicators.columns])),
    #                         indicators.shift(4).rename(columns=dict([(x, x + "_r4") for x in indicators.columns]))], axis=1))
    # plabel.append(pd.DataFrame({'p3': (candles.data.shift(-2).close / candles.data.shift(-1).open > 1.01) * 1}))
    plabel.append(cross.loc[observe.sum(axis=1) > 5, ['up', 'mid']])


five = pd.concat(start)
label = pd.concat(plabel)


dd = pd.concat([five, label], axis=1)
dd.to_csv('../data/6_in_9.csv')

# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(five.fillna(0).values, label.fillna(0).values)
# print(five.columns)
# print(clf.coef_)
# map(lambda x: print(x), zip(five.columns, clf.coef_))
#
# pred = clf.predict(five.fillna(0).values)
# label['pred'] = pred
#
# label.to_csv('../data/lasso_pred.csv')
#
# train = pd.concat([five, label], axis=1).fillna(0)
# splitter = BinSplitter(bad_name='p3',
#                        min_observation=1, min_target_observation=0,
#                        min_proportion=0, min_target_proportion=0, num_jobs=2)
# splitter.fit(train)
# splitter.apply(train[:1000])
# mrgr = BinMerger(min_drop=0, min_merge=0, min_merge_bad=0,
#                  min_iv=0, max_miss=0.999999,
#                  check_monotonicity=False, z_scale=False, num_jobs=2)
# mrgr.fit(splitter.data_summary)
#
# mrgr.save_variable_analysis('../data/var_analysis.txt')
# mrgr.save_woe_bins('../data/woe_bins.txt')
