import numpy as np
import pandas as pd
import QUANTAXIS as qa
from sklearn import linear_model
from functools import reduce


def skdj_buy(data):
    ind = pd.DataFrame({'SKDJ_BUY': pd.concat([data['SKDJ_K'] > data['SKDJ_D'],
                                               qa.REF(data['SKDJ_K'], 1) < qa.REF(data['SKDJ_D'], 1),
                                               qa.REF(data['SKDJ_D'], 1) < 20], axis=1).all(axis=1) * 1})
    return ind


def AVEDEV(Series, N):
    return Series.rolling(N).apply(lambda x: (np.abs(x - x.mean())).sum()/x.size)


def CCI(data, N=14):
    typ = (data['high'] + data['low'] + data['close']) / 3
    cci = ((typ - qa.MA(typ, N)) / (0.015 * AVEDEV(typ, N)))
    a = 100
    b = -100

    return pd.DataFrame({
        'CCI': cci, 'a': a, 'b': b
    })

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


labels = pd.read_csv('../data/cci_start.csv', dtype={'code': np.object}, parse_dates=['date'])

BACK_DAYS = 90
train = []
test = []
stocks = qa.QA_fetch_stock_list_adv()
stock_list = stocks.code.tolist()
for stock in stock_list:
    print('handling %s' % stock)
    try:
        candles = qa.QA_fetch_stock_day_adv(stock, start='2014-01-01', end='2018-05-24').to_qfq()
    except:
        print('data error during {}'.format(stock))
        continue

    try:
        skdj_9_3 = candles.add_func(qa.QA_indicator_SKDJ, N=9, M=3).rename(columns={'SKDJ_D': 'SKDJ_D_9_3', 'SKDJ_K': 'SKDJ_K_9_3'})
        cci_14 = candles.add_func(CCI, N=14)['CCI'].rename(columns={'CCI': 'CCI_14'})
        cci_83 = candles.add_func(CCI, N=83)['CCI'].rename(columns={'CCI': 'CCI_83'})
        macd_12_26 = candles.add_func(qa.QA_indicator_MACD, short=12, long=26, mid=9).rename(columns={'DIF': 'DIF_12_26', 'DEA': 'DEA_12_26', 'MACD': 'MACD_12_26'})
        roc_25 = candles.add_func(ROC, N=25).rename(columns={'ROC': 'ROC_25', 'MAROC': 'MAROC_25', 'EMAROC': 'EMAROC_25'})
        dtgf = candles.add_func(DTGF, N=21).rename(columns={'FAST': 'DTGF_FAST', 'SLOW': 'DTGF_SLOW'})
        bdzw = candles.add_func(BDZW).rename(columns={'FAST': 'BDZW_FAST', 'SLOW': 'BDZW_SLOW'})
        jmdd = candles.add_func(JMDD).rename(columns={'FAST': 'JMDD_FAST', 'SLOW': 'JMDD_SLOW'})
        mdz = candles.add_func(MDZ).rename(columns={'FAST': 'MDZ_FAST', 'SLOW': 'MDZ_SLOW'})
        lwr_9 = candles.add_func(LWR, N=9, M=3).rename(columns={'FAST': 'LWR_SLOW_9', 'SLOW': 'LWR_FAST_9'})

        boll_99 = candles.add_func(qa.QA_indicator_BOLL, N=99).rename(columns={'BOLL': 'BOLL_99', 'UB': 'UB_99', 'LB': 'LB_99'})
        boll_20 = candles.add_func(qa.QA_indicator_BOLL, N=20).rename(columns={'BOLL': 'BOLL_20', 'UB': 'UB_20', 'LB': 'LB_20'})
        boll_13 = candles.add_func(qa.QA_indicator_BOLL, N=13).rename(columns={'BOLL': 'BOLL_13', 'UB': 'UB_13', 'LB': 'LB_13'})
    except:
        print('func error during: {}'.format(stock))
        continue

    unquant = pd.concat([skdj_9_3, cci_14, cci_83, macd_12_26, roc_25, lwr_9, dtgf, bdzw, jmdd, mdz], axis=1)
    quant = pd.concat([candles.data.loc[:, ['open', 'close', 'high', 'low']], boll_13, boll_20, boll_99], axis=1)
    vol = pd.DataFrame(candles.vol)

    examples_unquant = pd.concat([unquant.shift(x).rename(columns=dict([(n, '{}_d{}'.format(n, x)) for n in unquant.columns])) for x in range(BACK_DAYS)], axis=1)
    examples_quant = pd.concat([quant.shift(x).rename(columns=dict([(n, '{}_d{}'.format(n, x)) for n in quant.columns])) for x in range(BACK_DAYS)], axis=1).div(quant.close, axis=0) - 1
    examples_vol = pd.concat([vol.shift(x).rename(columns=dict([(n, '{}_d{}'.format(n, x)) for n in vol.columns])) for x in range(BACK_DAYS)], axis=1).div(vol.volume, axis=0)

    example = pd.concat([examples_quant, examples_unquant, examples_vol], axis=1).fillna(0)
    # print(example.head())

    stock_label = labels.loc[labels.code == stock, ['date', 'code', 'GOUP']]
    # print(stock_label.head())

    full = pd.merge(example.reset_index(), stock_label, on=['date', 'code'])
    train.append(full.iloc[:-3, :])
    test.append(full.iloc[-3:, :])

data = pd.concat(train)
test = pd.concat(test)

# print(data.head(20))
data.to_hdf('../data/indi_train.hdf', 'data')
test.to_hdf('../data/indi_test.hdf', 'data')


clf = linear_model.Lasso(alpha=0.1)
clf.fit(data.iloc[:, 2: 3153].values, data.iloc[:, 3153].values)
for col, cof in zip(data.columns[2: 3153], clf.coef_):
    if cof != 0:
        print(col, cof)

pred = clf.predict(test.iloc[:, 2: 3153].values)
test['pred'] = pred

test.loc[:, ['date', 'code', 'GOUP', 'pred']].to_csv('../data/lasso_pred.csv', index=False)
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
