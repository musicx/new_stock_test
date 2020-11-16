import datetime
import numpy as np
import pandas as pd
import QUANTAXIS as qa
from sklearn import linear_model
from functools import reduce
from joblib import Parallel, delayed

from pymodline.labeled import forest_trainer as ft
from pymodline import model_loader as ml


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


def prepare(stock):
    BACK_DAYS = 90
    end = datetime.date.today()
    begin = end - datetime.timedelta(days=250)

    print('handling %s' % stock)
    try:
        candles = qa.QA_fetch_stock_day_adv(stock, start=str(begin), end=str(end)).to_qfq()
    except:
        print('data error during {}'.format(stock))
        return None

    try:
        skdj_9_3 = candles.add_func(qa.QA_indicator_SKDJ, N=9, M=3)
        cci_14 = candles.add_func(qa.QA_indicator_CCI, N=14)
        cci_83 = candles.add_func(qa.QA_indicator_CCI, N=83)
        macd_12_26 = candles.add_func(qa.QA_indicator_MACD, short=12, long=26, mid=9)
        roc_25 = candles.add_func(ROC, N=25)
        dtgf = candles.add_func(DTGF, N=21)
        bdzw = candles.add_func(BDZW)
        jmdd = candles.add_func(JMDD)
        mdz = candles.add_func(MDZ)
        lwr_9 = candles.add_func(LWR, N=9, M=3)

        boll_99 = candles.add_func(qa.QA_indicator_BOLL, N=99).rename(columns={'BOLL': 'BOLL_99', 'UB': 'UB_99', 'LB': 'LB_99'})
        boll_20 = candles.add_func(qa.QA_indicator_BOLL, N=20).rename(columns={'BOLL': 'BOLL_20', 'UB': 'UB_20', 'LB': 'LB_20'})
        boll_13 = candles.add_func(qa.QA_indicator_BOLL, N=13).rename(columns={'BOLL': 'BOLL_13', 'UB': 'UB_13', 'LB': 'LB_13'})

        buys = [skdj_buy(skdj_9_3),
                cci_buy(cci_14).rename(columns={'CCI_BUY': 'CCI_BUY_14_low'}),
                cci_buy(cci_14, 0).rename(columns={'CCI_BUY': 'CCI_BUY_14_mid'}),
                # cci_buy(cci_14, 100).rename(columns={'CCI_BUY': 'CCI_BUY_14_100'}),
                # cci_buy(cci_83).rename(columns={'CCI_BUY': 'CCI_BUY_83_m100'}),
                # cci_buy(cci_83, 0).rename(columns={'CCI_BUY': 'CCI_BUY_83_0'}),
                # cci_buy(cci_83, 100).rename(columns={'CCI_BUY': 'CCI_BUY_83_100'}),
                roc_buy(roc_25),
                normal_buy(dtgf).rename(columns={'NORM_BUY': 'DTGF_BUY'}),
                normal_buy(bdzw).rename(columns={'NORM_BUY': 'BDZW_BUY'}),
                normal_buy(jmdd).rename(columns={'NORM_BUY': 'JMDD_BUY'}),
                normal_buy(mdz).rename(columns={'NORM_BUY': 'MDZ_BUY'}),
                normal_buy(lwr_9).rename(columns={'NORM_BUY': 'LWR_BUY'}),
                macd_buy(macd_12_26)]

        indicators = pd.concat(buys, axis=1)
        observe = pd.concat([indicators, indicators.shift(1).rename(columns=dict([(x, x + "_r1") for x in indicators.columns]))], axis=1)
        if observe.sum(axis=1)[-1] < 6:
            return None

        skdj_9_3 = skdj_9_3.rename(columns={'SKDJ_D': 'SKDJ_D_9_3', 'SKDJ_K': 'SKDJ_K_9_3'})
        cci_14 = cci_14['CCI'].rename(columns={'CCI': 'CCI_14'})
        cci_83 = cci_83['CCI'].rename(columns={'CCI': 'CCI_83'})
        macd_12_26 = macd_12_26.rename(columns={'DIF': 'DIF_12_26', 'DEA': 'DEA_12_26', 'MACD': 'MACD_12_26'})
        roc_25 = roc_25.rename(columns={'ROC': 'ROC_25', 'MAROC': 'MAROC_25', 'EMAROC': 'EMAROC_25'})
        dtgf = dtgf.rename(columns={'FAST': 'DTGF_FAST', 'SLOW': 'DTGF_SLOW'})
        bdzw = bdzw.rename(columns={'FAST': 'BDZW_FAST', 'SLOW': 'BDZW_SLOW'})
        jmdd = jmdd.rename(columns={'FAST': 'JMDD_FAST', 'SLOW': 'JMDD_SLOW'})
        mdz = mdz.rename(columns={'FAST': 'MDZ_FAST', 'SLOW': 'MDZ_SLOW'})
        lwr_9 = lwr_9.rename(columns={'FAST': 'LWR_SLOW_9', 'SLOW': 'LWR_FAST_9'})

    except:
        print('func error during: {}'.format(stock))
        return None

    unquant = pd.concat([skdj_9_3, cci_14, cci_83, macd_12_26, roc_25, lwr_9, dtgf, bdzw, jmdd, mdz], axis=1)
    quant = pd.concat([candles.data.loc[:, ['open', 'close', 'high', 'low']], boll_13, boll_20, boll_99], axis=1)
    vol = pd.DataFrame(candles.vol)

    examples_unquant = pd.concat([unquant.shift(x).rename(columns=dict([(n, '{}_d{}'.format(n, x)) for n in unquant.columns])) for x in range(BACK_DAYS)], axis=1)
    examples_quant = pd.concat([quant.shift(x).rename(columns=dict([(n, '{}_d{}'.format(n, x)) for n in quant.columns])) for x in range(BACK_DAYS)], axis=1).div(quant.close, axis=0) - 1
    examples_vol = pd.concat([vol.shift(x).rename(columns=dict([(n, '{}_d{}'.format(n, x)) for n in vol.columns])) for x in range(BACK_DAYS)], axis=1).div(vol.volume, axis=0)

    example = pd.concat([examples_quant, examples_unquant, examples_vol], axis=1).fillna(0)

    return example.iloc[-1:, :]


if __name__ == "__main__":
    data = []
    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()
    # for stock in stock_list[:1]:
    #     data.append(prepare(stock))
    data = Parallel(n_jobs=4)(delayed(prepare)(stock) for stock in stock_list)
    if len(data) == 0:
        print('no valid stock')
        exit(0)

    candidates = pd.concat([x for x in data if x is not None])
    print('data prepared')

    classifier, names = ml.load_model('../models/random_forest_ind_T_1000_V_3150_D_20_MS_3_ML_2_R_206.pkl',
                                      '../features/random_forest_ind_T_1000_V_3150_D_20_MS_3_ML_2_R_206_feature.csv')
    score = ft.score(candidates.reset_index(), classifier, names, keep_columns=['date', 'code'])
    print('score calculated')

    today = datetime.date.today()
    score.drop(columns=['prob_of_0']).rename(columns={'prob_of_1': 'score'}).sort_values('score', ascending=False).to_csv('../data/rf_predict_{}.txt'.format(today), sep='\t', index=False)
    print('done')
