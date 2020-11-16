import numpy as np
import pandas as pd
import QUANTAXIS as qa
from functools import reduce
from scipy.stats import linregress


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


def STSLP(data, M=21, N=42):
    slope = data.close.rolling(M).apply(lambda x: linregress(range(M), x).slope)
    line = qa.EMA(slope * 20 + data.close, N)
    high = line * 1.10
    low = line * 0.9
    return pd.DataFrame({'SLPM': line, 'SLPU': high, 'SLPL': low})


def lowlatency(price, alpha):
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


def LLT(data, A=0.05):
    part = data.loc[:, ['open', 'close']]
    part['llt'] = lowlatency(part.close, A)
    return pd.DataFrame({'LLT': part.llt})


def BIAS(data, N=18):
    '乖离率'
    bias = (data.close - qa.MA(data.close, N)) / qa.MA(data.close, N) * 100
    return pd.DataFrame({'BIAS': bias})
