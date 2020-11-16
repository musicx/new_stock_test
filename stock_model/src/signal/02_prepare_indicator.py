import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from functools import reduce
from index import *
from random import shuffle


def ma_lines(data, col, N):
    ma = qa.MA(data[col], N) / data[col]
    ema = qa.EMA(data[col], N) / data[col]
    sma = qa.SMA(data[col], N) / data[col]
    ub = (ma + 2 * qa.STD(data[col], N)) / data[col]
    lb = qa.MAX((ma - 2 * qa.STD(data[col], N)), data['zero']) / data[col]
    return pd.DataFrame({'MA': ma, 'EMA': ema, 'SMA': sma, 'UB': ub, 'LB': lb})


def mtm_ind(data, col, N=12, M=6):
    mtm = (data[col] - qa.REF(data[col], N)) / data[col]
    mtmma = qa.MA(mtm, M)
    return pd.DataFrame({'MTM': mtm, 'MTMMA': mtmma})


def vol_ind(data, N):
    vr = qa.SUM(qa.IF(data.close > data.close.shift(1), data['volume'], data['zero']), N)/qa.SUM(qa.IF(data.close <= data.close.shift(1), data['volume'], data['zero']), N)*100
    vrsi = qa.SMA(qa.MAX(data['volume']-data['volume'].shift(1), data['zero']), N) / qa.SMA(qa.ABS(data['volume']-data['volume'].shift(1)), N)*100
    obv = qa.SUM(qa.IF(data.close > data.close.shift(1), data['volume'], qa.IF(data.close < data.close.shift(1), -data['volume'], data['zero'])), N)/10000

    return pd.DataFrame({'VR': vr, 'VRSI': vrsi, 'OBV': obv})


def pr_ind(data, N=26):
    OPEN = data.open
    HIGH = data.high
    LOW = data.low
    CLOSE = data.close
    MID = (HIGH+LOW+CLOSE)/3
    AR = qa.SUM(HIGH-OPEN, N) / qa.MAX(qa.SUM(OPEN-LOW, N), data['zero'] + 0.01) / CLOSE
    BR = qa.SUM(qa.MAX(data.zero, HIGH-qa.REF(CLOSE, 1)), N) / qa.MAX(qa.SUM(qa.MAX(data.zero, qa.REF(CLOSE, 1)-LOW), N), data['zero'] + 0.01) / CLOSE
    CR = qa.SUM(qa.MAX(data.zero, HIGH-qa.REF(MID, 1)), N) / qa.MAX(qa.SUM(qa.MAX(data.zero, qa.REF(MID, 1)-LOW), N), data['zero'] + 0.01) / CLOSE
    return pd.DataFrame({'AR': AR, 'BR': BR, 'CR': CR})


def asi_ind(data, M1=26, M2=10):
    CLOSE = data['close']
    HIGH = data['high']
    LOW = data['low']
    OPEN = data['open']
    LC = data['close'].shift(1)
    AA = qa.ABS(HIGH - LC)
    BB = qa.ABS(LOW-LC)
    CC = qa.ABS(HIGH - qa.REF(LOW, 1))
    DD = qa.ABS(LC - qa.REF(OPEN, 1))
    R = qa.IF((AA > BB) & (AA > CC), AA+BB/2+DD/4, qa.IF((BB > CC) & (BB > AA), BB+AA/2+DD/4, CC+DD/4))
    X = (CLOSE - LC + (CLOSE - OPEN) / 2 + LC - qa.REF(OPEN, 1))
    SI = 16*X/R*qa.MAX(AA, BB)
    ASI = qa.SUM(SI, M1)
    ASIT = qa.MA(ASI, M2)
    return pd.DataFrame({ 'ASI': ASI, 'ASIT': ASIT })


def dmi_ind(data, M1=14, M2=6):
    """
    趋向指标 DMI
    """
    HIGH = data.high
    LOW = data.low
    CLOSE = data.close

    TR = qa.SUM(qa.MAX(qa.MAX(HIGH-LOW, qa.ABS(HIGH-qa.REF(CLOSE, 1))), qa.ABS(LOW-qa.REF(CLOSE, 1))), M1)
    HD = HIGH-qa.REF(HIGH, 1)
    LD = qa.REF(LOW, 1)-LOW
    DMP = qa.SUM(qa.IF((HD > 0) & (HD > LD), HD, data['zero']), M1)
    DMM = qa.SUM(qa.IF((LD > 0) & (LD > HD), LD, data['zero']), M1)
    DI1 = DMP*100/TR
    DI2 = DMM*100/TR
    ADX = qa.MA(qa.ABS(DI2-DI1)/(DI1+DI2)*100, M2)
    ADXR = (ADX+qa.REF(ADX, M2))/2
    return pd.DataFrame({ 'DI1': DI1, 'DI2': DI2, 'ADX': ADX, 'ADXR': ADXR })


def adtm_ind(data, N=23, M=8):
    HIGH_DIFF = data.high - data.open
    LOW_DIFF = data.open - data.low
    OPEN_DIFF = data.open - data.open.shift(1)
    DTM = qa.IF(data.open <= data.open.shift(1), data['zero'], qa.MAX(HIGH_DIFF, OPEN_DIFF))
    DBM = qa.IF(data.open >= data.open.shift(1), data['zero'], qa.MAX(LOW_DIFF, OPEN_DIFF))
    STM = qa.SUM(DTM, N)
    SBM = qa.SUM(DBM, N)
    ADTM1 = qa.IF(STM > SBM, (STM - SBM) / STM, qa.IF(STM == SBM, data['zero'], (STM - SBM) / SBM))
    MAADTM = qa.MA(ADTM1, M)
    DICT = {'ADTM': ADTM1, 'MAADTM': MAADTM}
    return pd.DataFrame(DICT)


def go_up(data, N=5, P=3, M=0.1):
    ind = (data.close > data.close.shift(1)) * 1
    checked = check(ind, -N)
    lift = data.high.rolling(N).max().shift(-N-1) / data.open.shift(-1)
    flat = ((data.close == data.open) & (data.close == data.high)) * 1
    return pd.DataFrame({'go_up': (~(flat > 0) & ~(flat.shift(-1) > 0) & (checked >= P) & (lift > (1+M))) * 1})


def check(signal, days):
    return reduce(lambda x, y: x + y, [signal.shift(x).fillna(0)
                                       for x in (range(days) if days >= 0 else range(-1, days-1, -1))])


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    today_str = today.strftime('%Y-%m-%d')

    # divide_date = today - dt.timedelta(days=60)
    # divide_str = divide_date.strftime('%Y-%m-%d')

    # stocks = qa.QA_fetch_stock_list_adv()
    # stock_list = stocks.code.tolist()
    stock_list = ZZ800.split('\n')
    shuffle(stock_list)
    for stock in stock_list[:10]:
        print('%s: handling %s' % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stock))
        try:
            candles = qa.QA_fetch_stock_day_adv(stock, start='2016-01-01', end=today_str).to_qfq()
            if candles.data.shape[0] <= 150:
                continue
        except:
            print('data error during {}'.format(stock))
            continue

        data = candles.data
        data['c1'] = (data['close'] + data['high'] + data['low']) / 3.
        data['c2'] = (data['close']*3 + data['high'] + data['low'] + data['open']) / 6.
        data['c3'] = data['amount'] / data['volume'] / 100.
        data['cho'] = qa.SUM(data['volume']*(2*data['close']-data['high']-data['low'])/(data['high'] + data['low']), 100)
        data['zero'] = 0

        # labels
        label = candles.add_func(go_up).loc[:, 'go_up']

        # quantiles
        pr = [candles.add_func(pr_ind, N=x).loc[:, ['AR', 'BR', 'CR']].rename(columns={'AR': 'ar_{}'.format(x), 'BR': 'br_{}'.format(x), 'CR': 'cr_{}'.format(x)})
              for x in range(3, 61, 3)]

        asi = [candles.add_func(asi_ind, M1=x, M2=y).loc[:, ['ASI', 'ASIT']].rename(columns={'ASI': 'asi_{}_{}'.format(x, y), 'ASIT': 'asit_{}_{}'.format(x, y)})
               for x in range(6, 31, 3) for y in range(3, 31, 3)]

        cci = [candles.add_func(qa.QA_indicator_CCI, N=x).loc[:, 'CCI'].rename(columns={'CCI': 'cci_{}'.format(x)})
               for x in range(6, 101, 2)]

        macd = [candles.add_func(qa.QA_indicator_MACD, short=x, long=y, mid=z).loc[:, ['DIF', 'DEA', 'MACD']].rename(columns={'DIF': 'dif_{}_{}_{}'.format(x, y, z),
                                                                                                                              'DEA': 'dea_{}_{}_{}'.format(x, y, z),
                                                                                                                              'MACD': 'macd_{}_{}_{}'.format(x, y, z)})
                for x in range(3, 31, 3) for y in range(4, 61, 2) for z in range(3, 31, 3) if x != y]

        kdj = [candles.add_func(qa.QA_indicator_KDJ, N=x, M1=y, M2=z).loc[:, ['KDJ_K', 'KDJ_D', 'KDJ_J']].rename(columns={'KDJ_K': 'kdj_k_{}_{}_{}'.format(x, y, z),
                                                                                                                          'KDJ_D': 'kdj_d_{}_{}_{}'.format(x, y, z),
                                                                                                                          'KDJ_J': 'kdj_j_{}_{}_{}'.format(x, y, z)})
               for x in range(3, 31, 3) for y in range(2, 21, 2) for z in range(2, 21, 2)]

        skdj = [candles.add_func(qa.QA_indicator_SKDJ, N=x, M=y).loc[:, ['RSV', 'SKDJ_K', 'SKDJ_D']].rename(columns={'RSV': 'skdj_v_{}_{}'.format(x, y), 'SKDJ_D': 'skdj_d_{}_{}'.format(x, y), 'SKDJ_K': 'skdj_k_{}_{}'.format(x, y)})
                for x in range(5, 101, 5) for y in range(2, 31, 2)]

        dmi = [candles.add_func(dmi_ind, M1=x, M2=y).loc[:, ['DI1', 'DI2', 'ADX', 'ADXR']].rename(columns={'DI1': 'dmi_dip_{}_{}'.format(x, y), 'DI2': 'dmi_dim_{}_{}'.format(x, y), 'ADX': 'dmi_adx_{}_{}'.format(x, y), 'ADXR': 'dmi_adxr_{}_{}'.format(x, y)})
               for x in range(6, 31, 3) for y in range(2, 10)]

        adtm = [candles.add_func(adtm_ind, N=x, M=y).loc[:, ['ADTM', 'MAADTM']].rename(columns={'ADTM': 'adtm_{}_{}'.format(x, y), 'MAADTM': 'adtm_ma_{}_{}'.format(x, y)})
                for x in range(3, 31, 3) for y in range(2, 51, 2)]

        vols = [candles.add_func(vol_ind, N=x).loc[:, ['VR', 'VRSI', 'OBV']].rename(columns={'VR': 'vr_{}'.format(x), 'VRSI': 'vrsi_{}'.format(x), 'OBV': 'obv_{}'.format(x)})
                for x in range(3, 101, 3)]

        # price lines, will be close / line
        mtm = [candles.add_func(mtm_ind, col='close', N=x, M=y).loc[:, ['MTM', 'MTMMA']].rename(columns={'MTM': 'mtm_{}_{}'.format(x, y), 'MTMMA': 'mtmma_{}_{}'.format(x, y)})
               for x in range(3, 31, 3) for y in range(2, 21, 2)]

        ma = [candles.add_func(ma_lines, col='close', N=3).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_3', 'EMA': 'ema_3', 'SMA': 'sma_3', 'UB': 'boll_ub_3', 'LB': 'boll_lb_3'}),
              candles.add_func(ma_lines, col='close', N=5).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_5', 'EMA': 'ema_5', 'SMA': 'sma_5', 'UB': 'boll_ub_5', 'LB': 'boll_lb_5'}),
              candles.add_func(ma_lines, col='close', N=8).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_8', 'EMA': 'ema_8', 'SMA': 'sma_8', 'UB': 'boll_ub_8', 'LB': 'boll_lb_8'}),
              candles.add_func(ma_lines, col='close', N=10).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_10', 'EMA': 'ema_10', 'SMA': 'sma_10', 'UB': 'boll_ub_10', 'LB': 'boll_lb_10'}),
              candles.add_func(ma_lines, col='close', N=13).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_13', 'EMA': 'ema_13', 'SMA': 'sma_13', 'UB': 'boll_ub_13', 'LB': 'boll_lb_13'}),
              candles.add_func(ma_lines, col='close', N=15).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_15', 'EMA': 'ema_15', 'SMA': 'sma_15', 'UB': 'boll_ub_15', 'LB': 'boll_lb_15'}),
              candles.add_func(ma_lines, col='close', N=18).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_18', 'EMA': 'ema_18', 'SMA': 'sma_18', 'UB': 'boll_ub_18', 'LB': 'boll_lb_18'}),
              candles.add_func(ma_lines, col='close', N=20).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_20', 'EMA': 'ema_20', 'SMA': 'sma_20', 'UB': 'boll_ub_20', 'LB': 'boll_lb_20'}),
              candles.add_func(ma_lines, col='close', N=21).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_21', 'EMA': 'ema_21', 'SMA': 'sma_21', 'UB': 'boll_ub_21', 'LB': 'boll_lb_21'}),
              candles.add_func(ma_lines, col='close', N=26).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_26', 'EMA': 'ema_26', 'SMA': 'sma_26', 'UB': 'boll_ub_26', 'LB': 'boll_lb_26'}),
              candles.add_func(ma_lines, col='close', N=30).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_30', 'EMA': 'ema_30', 'SMA': 'sma_30', 'UB': 'boll_ub_30', 'LB': 'boll_lb_30'}),
              candles.add_func(ma_lines, col='close', N=34).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_34', 'EMA': 'ema_34', 'SMA': 'sma_34', 'UB': 'boll_ub_34', 'LB': 'boll_lb_34'}),
              candles.add_func(ma_lines, col='close', N=40).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_40', 'EMA': 'ema_40', 'SMA': 'sma_40', 'UB': 'boll_ub_40', 'LB': 'boll_lb_40'}),
              candles.add_func(ma_lines, col='close', N=44).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_44', 'EMA': 'ema_44', 'SMA': 'sma_44', 'UB': 'boll_ub_44', 'LB': 'boll_lb_44'}),
              candles.add_func(ma_lines, col='close', N=50).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_50', 'EMA': 'ema_50', 'SMA': 'sma_50', 'UB': 'boll_ub_50', 'LB': 'boll_lb_50'}),
              candles.add_func(ma_lines, col='close', N=55).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_55', 'EMA': 'ema_55', 'SMA': 'sma_55', 'UB': 'boll_ub_55', 'LB': 'boll_lb_55'}),
              candles.add_func(ma_lines, col='close', N=60).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_60', 'EMA': 'ema_60', 'SMA': 'sma_60', 'UB': 'boll_ub_60', 'LB': 'boll_lb_60'}),
              candles.add_func(ma_lines, col='close', N=66).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_66', 'EMA': 'ema_66', 'SMA': 'sma_66', 'UB': 'boll_ub_66', 'LB': 'boll_lb_66'}),
              candles.add_func(ma_lines, col='close', N=89).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_89', 'EMA': 'ema_89', 'SMA': 'sma_89', 'UB': 'boll_ub_89', 'LB': 'boll_lb_89'}),
              candles.add_func(ma_lines, col='close', N=99).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_99', 'EMA': 'ema_99', 'SMA': 'sma_99', 'UB': 'boll_ub_99', 'LB': 'boll_lb_99'}),
              candles.add_func(ma_lines, col='close', N=120).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_120', 'EMA': 'ema_120', 'SMA': 'sma_120', 'UB': 'boll_ub_120', 'LB': 'boll_lb_120'}),
              candles.add_func(ma_lines, col='close', N=144).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_144', 'EMA': 'ema_144', 'SMA': 'sma_144', 'UB': 'boll_ub_144', 'LB': 'boll_lb_144'})]

        vma = [candles.add_func(ma_lines, col='volume', N=3).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_3v', 'EMA': 'ema_3v', 'SMA': 'sma_3v', 'UB': 'boll_ub_3v', 'LB': 'boll_lb_3v'}),
               candles.add_func(ma_lines, col='volume', N=5).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_5v', 'EMA': 'ema_5v', 'SMA': 'sma_5v', 'UB': 'boll_ub_5v', 'LB': 'boll_lb_5v'}),
               candles.add_func(ma_lines, col='volume', N=8).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_8v', 'EMA': 'ema_8v', 'SMA': 'sma_8v', 'UB': 'boll_ub_8v', 'LB': 'boll_lb_8v'}),
               candles.add_func(ma_lines, col='volume', N=10).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_10v', 'EMA': 'ema_10v', 'SMA': 'sma_10v', 'UB': 'boll_ub_10v', 'LB': 'boll_lb_10v'}),
               candles.add_func(ma_lines, col='volume', N=13).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_13v', 'EMA': 'ema_13v', 'SMA': 'sma_13v', 'UB': 'boll_ub_13v', 'LB': 'boll_lb_13v'}),
               candles.add_func(ma_lines, col='volume', N=15).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_15v', 'EMA': 'ema_15v', 'SMA': 'sma_15v', 'UB': 'boll_ub_15v', 'LB': 'boll_lb_15v'}),
               candles.add_func(ma_lines, col='volume', N=18).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_18v', 'EMA': 'ema_18v', 'SMA': 'sma_18v', 'UB': 'boll_ub_18v', 'LB': 'boll_lb_18v'}),
               candles.add_func(ma_lines, col='volume', N=20).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_20v', 'EMA': 'ema_20v', 'SMA': 'sma_20v', 'UB': 'boll_ub_20v', 'LB': 'boll_lb_20v'}),
               candles.add_func(ma_lines, col='volume', N=21).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_21v', 'EMA': 'ema_21v', 'SMA': 'sma_21v', 'UB': 'boll_ub_21v', 'LB': 'boll_lb_21v'}),
               candles.add_func(ma_lines, col='volume', N=26).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_26v', 'EMA': 'ema_26v', 'SMA': 'sma_26v', 'UB': 'boll_ub_26v', 'LB': 'boll_lb_26v'}),
               candles.add_func(ma_lines, col='volume', N=30).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_30v', 'EMA': 'ema_30v', 'SMA': 'sma_30v', 'UB': 'boll_ub_30v', 'LB': 'boll_lb_30v'}),
               candles.add_func(ma_lines, col='volume', N=34).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_34v', 'EMA': 'ema_34v', 'SMA': 'sma_34v', 'UB': 'boll_ub_34v', 'LB': 'boll_lb_34v'}),
               candles.add_func(ma_lines, col='volume', N=40).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_40v', 'EMA': 'ema_40v', 'SMA': 'sma_40v', 'UB': 'boll_ub_40v', 'LB': 'boll_lb_40v'}),
               candles.add_func(ma_lines, col='volume', N=44).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_44v', 'EMA': 'ema_44v', 'SMA': 'sma_44v', 'UB': 'boll_ub_44v', 'LB': 'boll_lb_44v'}),
               candles.add_func(ma_lines, col='volume', N=50).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_50v', 'EMA': 'ema_50v', 'SMA': 'sma_50v', 'UB': 'boll_ub_50v', 'LB': 'boll_lb_50v'}),
               candles.add_func(ma_lines, col='volume', N=55).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_55v', 'EMA': 'ema_55v', 'SMA': 'sma_55v', 'UB': 'boll_ub_55v', 'LB': 'boll_lb_55v'}),
               candles.add_func(ma_lines, col='volume', N=60).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_60v', 'EMA': 'ema_60v', 'SMA': 'sma_60v', 'UB': 'boll_ub_60v', 'LB': 'boll_lb_60v'}),
               candles.add_func(ma_lines, col='volume', N=66).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_66v', 'EMA': 'ema_66v', 'SMA': 'sma_66v', 'UB': 'boll_ub_66v', 'LB': 'boll_lb_66v'}),
               candles.add_func(ma_lines, col='volume', N=89).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_89v', 'EMA': 'ema_89v', 'SMA': 'sma_89v', 'UB': 'boll_ub_89v', 'LB': 'boll_lb_89v'}),
               candles.add_func(ma_lines, col='volume', N=99).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_99v', 'EMA': 'ema_99v', 'SMA': 'sma_99v', 'UB': 'boll_ub_99v', 'LB': 'boll_lb_99v'}),
               candles.add_func(ma_lines, col='volume', N=120).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_120v', 'EMA': 'ema_120v', 'SMA': 'sma_120v', 'UB': 'boll_ub_120v', 'LB': 'boll_lb_120v'}),
               candles.add_func(ma_lines, col='volume', N=144).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_144v', 'EMA': 'ema_144v', 'SMA': 'sma_144v', 'UB': 'boll_ub_144v', 'LB': 'boll_lb_144v'})]

        ma1 = [candles.add_func(ma_lines, col='c1', N=3).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_3_c1', 'EMA': 'ema_3_c1', 'SMA': 'sma_3_c1', 'UB': 'boll_ub_3_c1', 'LB': 'boll_lb_3_c1'}),
               candles.add_func(ma_lines, col='c1', N=5).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_5_c1', 'EMA': 'ema_5_c1', 'SMA': 'sma_5_c1', 'UB': 'boll_ub_5_c1', 'LB': 'boll_lb_5_c1'}),
               candles.add_func(ma_lines, col='c1', N=8).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_8_c1', 'EMA': 'ema_8_c1', 'SMA': 'sma_8_c1', 'UB': 'boll_ub_8_c1', 'LB': 'boll_lb_8_c1'}),
               candles.add_func(ma_lines, col='c1', N=10).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_10_c1', 'EMA': 'ema_10_c1', 'SMA': 'sma_10_c1', 'UB': 'boll_ub_10_c1', 'LB': 'boll_lb_10_c1'}),
               candles.add_func(ma_lines, col='c1', N=13).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_13_c1', 'EMA': 'ema_13_c1', 'SMA': 'sma_13_c1', 'UB': 'boll_ub_13_c1', 'LB': 'boll_lb_13_c1'}),
               candles.add_func(ma_lines, col='c1', N=15).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_15_c1', 'EMA': 'ema_15_c1', 'SMA': 'sma_15_c1', 'UB': 'boll_ub_15_c1', 'LB': 'boll_lb_15_c1'}),
               candles.add_func(ma_lines, col='c1', N=18).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_18_c1', 'EMA': 'ema_18_c1', 'SMA': 'sma_18_c1', 'UB': 'boll_ub_18_c1', 'LB': 'boll_lb_18_c1'}),
               candles.add_func(ma_lines, col='c1', N=20).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_20_c1', 'EMA': 'ema_20_c1', 'SMA': 'sma_20_c1', 'UB': 'boll_ub_20_c1', 'LB': 'boll_lb_20_c1'}),
               candles.add_func(ma_lines, col='c1', N=21).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_21_c1', 'EMA': 'ema_21_c1', 'SMA': 'sma_21_c1', 'UB': 'boll_ub_21_c1', 'LB': 'boll_lb_21_c1'}),
               candles.add_func(ma_lines, col='c1', N=26).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_26_c1', 'EMA': 'ema_26_c1', 'SMA': 'sma_26_c1', 'UB': 'boll_ub_26_c1', 'LB': 'boll_lb_26_c1'}),
               candles.add_func(ma_lines, col='c1', N=30).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_30_c1', 'EMA': 'ema_30_c1', 'SMA': 'sma_30_c1', 'UB': 'boll_ub_30_c1', 'LB': 'boll_lb_30_c1'}),
               candles.add_func(ma_lines, col='c1', N=34).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_34_c1', 'EMA': 'ema_34_c1', 'SMA': 'sma_34_c1', 'UB': 'boll_ub_34_c1', 'LB': 'boll_lb_34_c1'}),
               candles.add_func(ma_lines, col='c1', N=40).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_40_c1', 'EMA': 'ema_40_c1', 'SMA': 'sma_40_c1', 'UB': 'boll_ub_40_c1', 'LB': 'boll_lb_40_c1'}),
               candles.add_func(ma_lines, col='c1', N=44).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_44_c1', 'EMA': 'ema_44_c1', 'SMA': 'sma_44_c1', 'UB': 'boll_ub_44_c1', 'LB': 'boll_lb_44_c1'}),
               candles.add_func(ma_lines, col='c1', N=50).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_50_c1', 'EMA': 'ema_50_c1', 'SMA': 'sma_50_c1', 'UB': 'boll_ub_50_c1', 'LB': 'boll_lb_50_c1'}),
               candles.add_func(ma_lines, col='c1', N=55).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_55_c1', 'EMA': 'ema_55_c1', 'SMA': 'sma_55_c1', 'UB': 'boll_ub_55_c1', 'LB': 'boll_lb_55_c1'}),
               candles.add_func(ma_lines, col='c1', N=60).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_60_c1', 'EMA': 'ema_60_c1', 'SMA': 'sma_60_c1', 'UB': 'boll_ub_60_c1', 'LB': 'boll_lb_60_c1'}),
               candles.add_func(ma_lines, col='c1', N=66).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_66_c1', 'EMA': 'ema_66_c1', 'SMA': 'sma_66_c1', 'UB': 'boll_ub_66_c1', 'LB': 'boll_lb_66_c1'}),
               candles.add_func(ma_lines, col='c1', N=89).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_89_c1', 'EMA': 'ema_89_c1', 'SMA': 'sma_89_c1', 'UB': 'boll_ub_89_c1', 'LB': 'boll_lb_89_c1'}),
               candles.add_func(ma_lines, col='c1', N=99).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_99_c1', 'EMA': 'ema_99_c1', 'SMA': 'sma_99_c1', 'UB': 'boll_ub_99_c1', 'LB': 'boll_lb_99_c1'}),
               candles.add_func(ma_lines, col='c1', N=120).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_120_c1', 'EMA': 'ema_120_c1', 'SMA': 'sma_120_c1', 'UB': 'boll_ub_120_c1', 'LB': 'boll_lb_120_c1'}),
               candles.add_func(ma_lines, col='c1', N=144).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_144_c1', 'EMA': 'ema_144_c1', 'SMA': 'sma_144_c1', 'UB': 'boll_ub_144_c1', 'LB': 'boll_lb_144_c1'})]

        ma2 = [candles.add_func(ma_lines, col='c2', N=3).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_3_c2', 'EMA': 'ema_3_c2', 'SMA': 'sma_3_c2', 'UB': 'boll_ub_3_c2', 'LB': 'boll_lb_3_c2'}),
               candles.add_func(ma_lines, col='c2', N=5).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_5_c2', 'EMA': 'ema_5_c2', 'SMA': 'sma_5_c2', 'UB': 'boll_ub_5_c2', 'LB': 'boll_lb_5_c2'}),
               candles.add_func(ma_lines, col='c2', N=8).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_8_c2', 'EMA': 'ema_8_c2', 'SMA': 'sma_8_c2', 'UB': 'boll_ub_8_c2', 'LB': 'boll_lb_8_c2'}),
               candles.add_func(ma_lines, col='c2', N=10).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_10_c2', 'EMA': 'ema_10_c2', 'SMA': 'sma_10_c2', 'UB': 'boll_ub_10_c2', 'LB': 'boll_lb_10_c2'}),
               candles.add_func(ma_lines, col='c2', N=13).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_13_c2', 'EMA': 'ema_13_c2', 'SMA': 'sma_13_c2', 'UB': 'boll_ub_13_c2', 'LB': 'boll_lb_13_c2'}),
               candles.add_func(ma_lines, col='c2', N=15).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_15_c2', 'EMA': 'ema_15_c2', 'SMA': 'sma_15_c2', 'UB': 'boll_ub_15_c2', 'LB': 'boll_lb_15_c2'}),
               candles.add_func(ma_lines, col='c2', N=18).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_18_c2', 'EMA': 'ema_18_c2', 'SMA': 'sma_18_c2', 'UB': 'boll_ub_18_c2', 'LB': 'boll_lb_18_c2'}),
               candles.add_func(ma_lines, col='c2', N=20).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_20_c2', 'EMA': 'ema_20_c2', 'SMA': 'sma_20_c2', 'UB': 'boll_ub_20_c2', 'LB': 'boll_lb_20_c2'}),
               candles.add_func(ma_lines, col='c2', N=21).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_21_c2', 'EMA': 'ema_21_c2', 'SMA': 'sma_21_c2', 'UB': 'boll_ub_21_c2', 'LB': 'boll_lb_21_c2'}),
               candles.add_func(ma_lines, col='c2', N=26).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_26_c2', 'EMA': 'ema_26_c2', 'SMA': 'sma_26_c2', 'UB': 'boll_ub_26_c2', 'LB': 'boll_lb_26_c2'}),
               candles.add_func(ma_lines, col='c2', N=30).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_30_c2', 'EMA': 'ema_30_c2', 'SMA': 'sma_30_c2', 'UB': 'boll_ub_30_c2', 'LB': 'boll_lb_30_c2'}),
               candles.add_func(ma_lines, col='c2', N=34).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_34_c2', 'EMA': 'ema_34_c2', 'SMA': 'sma_34_c2', 'UB': 'boll_ub_34_c2', 'LB': 'boll_lb_34_c2'}),
               candles.add_func(ma_lines, col='c2', N=40).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_40_c2', 'EMA': 'ema_40_c2', 'SMA': 'sma_40_c2', 'UB': 'boll_ub_40_c2', 'LB': 'boll_lb_40_c2'}),
               candles.add_func(ma_lines, col='c2', N=44).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_44_c2', 'EMA': 'ema_44_c2', 'SMA': 'sma_44_c2', 'UB': 'boll_ub_44_c2', 'LB': 'boll_lb_44_c2'}),
               candles.add_func(ma_lines, col='c2', N=50).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_50_c2', 'EMA': 'ema_50_c2', 'SMA': 'sma_50_c2', 'UB': 'boll_ub_50_c2', 'LB': 'boll_lb_50_c2'}),
               candles.add_func(ma_lines, col='c2', N=55).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_55_c2', 'EMA': 'ema_55_c2', 'SMA': 'sma_55_c2', 'UB': 'boll_ub_55_c2', 'LB': 'boll_lb_55_c2'}),
               candles.add_func(ma_lines, col='c2', N=60).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_60_c2', 'EMA': 'ema_60_c2', 'SMA': 'sma_60_c2', 'UB': 'boll_ub_60_c2', 'LB': 'boll_lb_60_c2'}),
               candles.add_func(ma_lines, col='c2', N=66).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_66_c2', 'EMA': 'ema_66_c2', 'SMA': 'sma_66_c2', 'UB': 'boll_ub_66_c2', 'LB': 'boll_lb_66_c2'}),
               candles.add_func(ma_lines, col='c2', N=89).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_89_c2', 'EMA': 'ema_89_c2', 'SMA': 'sma_89_c2', 'UB': 'boll_ub_89_c2', 'LB': 'boll_lb_89_c2'}),
               candles.add_func(ma_lines, col='c2', N=99).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_99_c2', 'EMA': 'ema_99_c2', 'SMA': 'sma_99_c2', 'UB': 'boll_ub_99_c2', 'LB': 'boll_lb_99_c2'}),
               candles.add_func(ma_lines, col='c2', N=120).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_120_c2', 'EMA': 'ema_120_c2', 'SMA': 'sma_120_c2', 'UB': 'boll_ub_120_c2', 'LB': 'boll_lb_120_c2'}),
               candles.add_func(ma_lines, col='c2', N=144).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_144_c2', 'EMA': 'ema_144_c2', 'SMA': 'sma_144_c2', 'UB': 'boll_ub_144_c2', 'LB': 'boll_lb_144_c2'})]

        ma3 = [candles.add_func(ma_lines, col='c3', N=3).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_3_c3', 'EMA': 'ema_3_c3', 'SMA': 'sma_3_c3', 'UB': 'boll_ub_3_c3', 'LB': 'boll_lb_3_c3'}),
               candles.add_func(ma_lines, col='c3', N=5).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_5_c3', 'EMA': 'ema_5_c3', 'SMA': 'sma_5_c3', 'UB': 'boll_ub_5_c3', 'LB': 'boll_lb_5_c3'}),
               candles.add_func(ma_lines, col='c3', N=8).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_8_c3', 'EMA': 'ema_8_c3', 'SMA': 'sma_8_c3', 'UB': 'boll_ub_8_c3', 'LB': 'boll_lb_8_c3'}),
               candles.add_func(ma_lines, col='c3', N=10).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_10_c3', 'EMA': 'ema_10_c3', 'SMA': 'sma_10_c3', 'UB': 'boll_ub_10_c3', 'LB': 'boll_lb_10_c3'}),
               candles.add_func(ma_lines, col='c3', N=13).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_13_c3', 'EMA': 'ema_13_c3', 'SMA': 'sma_13_c3', 'UB': 'boll_ub_13_c3', 'LB': 'boll_lb_13_c3'}),
               candles.add_func(ma_lines, col='c3', N=15).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_15_c3', 'EMA': 'ema_15_c3', 'SMA': 'sma_15_c3', 'UB': 'boll_ub_15_c3', 'LB': 'boll_lb_15_c3'}),
               candles.add_func(ma_lines, col='c3', N=18).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_18_c3', 'EMA': 'ema_18_c3', 'SMA': 'sma_18_c3', 'UB': 'boll_ub_18_c3', 'LB': 'boll_lb_18_c3'}),
               candles.add_func(ma_lines, col='c3', N=20).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_20_c3', 'EMA': 'ema_20_c3', 'SMA': 'sma_20_c3', 'UB': 'boll_ub_20_c3', 'LB': 'boll_lb_20_c3'}),
               candles.add_func(ma_lines, col='c3', N=21).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_21_c3', 'EMA': 'ema_21_c3', 'SMA': 'sma_21_c3', 'UB': 'boll_ub_21_c3', 'LB': 'boll_lb_21_c3'}),
               candles.add_func(ma_lines, col='c3', N=26).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_26_c3', 'EMA': 'ema_26_c3', 'SMA': 'sma_26_c3', 'UB': 'boll_ub_26_c3', 'LB': 'boll_lb_26_c3'}),
               candles.add_func(ma_lines, col='c3', N=30).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_30_c3', 'EMA': 'ema_30_c3', 'SMA': 'sma_30_c3', 'UB': 'boll_ub_30_c3', 'LB': 'boll_lb_30_c3'}),
               candles.add_func(ma_lines, col='c3', N=34).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_34_c3', 'EMA': 'ema_34_c3', 'SMA': 'sma_34_c3', 'UB': 'boll_ub_34_c3', 'LB': 'boll_lb_34_c3'}),
               candles.add_func(ma_lines, col='c3', N=40).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_40_c3', 'EMA': 'ema_40_c3', 'SMA': 'sma_40_c3', 'UB': 'boll_ub_40_c3', 'LB': 'boll_lb_40_c3'}),
               candles.add_func(ma_lines, col='c3', N=44).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_44_c3', 'EMA': 'ema_44_c3', 'SMA': 'sma_44_c3', 'UB': 'boll_ub_44_c3', 'LB': 'boll_lb_44_c3'}),
               candles.add_func(ma_lines, col='c3', N=50).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_50_c3', 'EMA': 'ema_50_c3', 'SMA': 'sma_50_c3', 'UB': 'boll_ub_50_c3', 'LB': 'boll_lb_50_c3'}),
               candles.add_func(ma_lines, col='c3', N=55).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_55_c3', 'EMA': 'ema_55_c3', 'SMA': 'sma_55_c3', 'UB': 'boll_ub_55_c3', 'LB': 'boll_lb_55_c3'}),
               candles.add_func(ma_lines, col='c3', N=60).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_60_c3', 'EMA': 'ema_60_c3', 'SMA': 'sma_60_c3', 'UB': 'boll_ub_60_c3', 'LB': 'boll_lb_60_c3'}),
               candles.add_func(ma_lines, col='c3', N=66).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_66_c3', 'EMA': 'ema_66_c3', 'SMA': 'sma_66_c3', 'UB': 'boll_ub_66_c3', 'LB': 'boll_lb_66_c3'}),
               candles.add_func(ma_lines, col='c3', N=89).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_89_c3', 'EMA': 'ema_89_c3', 'SMA': 'sma_89_c3', 'UB': 'boll_ub_89_c3', 'LB': 'boll_lb_89_c3'}),
               candles.add_func(ma_lines, col='c3', N=99).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_99_c3', 'EMA': 'ema_99_c3', 'SMA': 'sma_99_c3', 'UB': 'boll_ub_99_c3', 'LB': 'boll_lb_99_c3'}),
               candles.add_func(ma_lines, col='c3', N=120).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_120_c3', 'EMA': 'ema_120_c3', 'SMA': 'sma_120_c3', 'UB': 'boll_ub_120_c3', 'LB': 'boll_lb_120_c3'}),
               candles.add_func(ma_lines, col='c3', N=144).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_144_c3', 'EMA': 'ema_144_c3', 'SMA': 'sma_144_c3', 'UB': 'boll_ub_144_c3', 'LB': 'boll_lb_144_c3'})]

        macho = [candles.add_func(ma_lines, col='cho', N=3).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_3_cho', 'EMA': 'ema_3_cho', 'SMA': 'sma_3_cho', 'UB': 'boll_ub_3_cho', 'LB': 'boll_lb_3_cho'}),
                 candles.add_func(ma_lines, col='cho', N=5).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_5_cho', 'EMA': 'ema_5_cho', 'SMA': 'sma_5_cho', 'UB': 'boll_ub_5_cho', 'LB': 'boll_lb_5_cho'}),
                 candles.add_func(ma_lines, col='cho', N=8).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_8_cho', 'EMA': 'ema_8_cho', 'SMA': 'sma_8_cho', 'UB': 'boll_ub_8_cho', 'LB': 'boll_lb_8_cho'}),
                 candles.add_func(ma_lines, col='cho', N=10).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_10_cho', 'EMA': 'ema_10_cho', 'SMA': 'sma_10_cho', 'UB': 'boll_ub_10_cho', 'LB': 'boll_lb_10_cho'}),
                 candles.add_func(ma_lines, col='cho', N=13).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_13_cho', 'EMA': 'ema_13_cho', 'SMA': 'sma_13_cho', 'UB': 'boll_ub_13_cho', 'LB': 'boll_lb_13_cho'}),
                 candles.add_func(ma_lines, col='cho', N=15).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_15_cho', 'EMA': 'ema_15_cho', 'SMA': 'sma_15_cho', 'UB': 'boll_ub_15_cho', 'LB': 'boll_lb_15_cho'}),
                 candles.add_func(ma_lines, col='cho', N=18).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_18_cho', 'EMA': 'ema_18_cho', 'SMA': 'sma_18_cho', 'UB': 'boll_ub_18_cho', 'LB': 'boll_lb_18_cho'}),
                 candles.add_func(ma_lines, col='cho', N=20).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_20_cho', 'EMA': 'ema_20_cho', 'SMA': 'sma_20_cho', 'UB': 'boll_ub_20_cho', 'LB': 'boll_lb_20_cho'}),
                 candles.add_func(ma_lines, col='cho', N=21).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_21_cho', 'EMA': 'ema_21_cho', 'SMA': 'sma_21_cho', 'UB': 'boll_ub_21_cho', 'LB': 'boll_lb_21_cho'}),
                 candles.add_func(ma_lines, col='cho', N=26).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_26_cho', 'EMA': 'ema_26_cho', 'SMA': 'sma_26_cho', 'UB': 'boll_ub_26_cho', 'LB': 'boll_lb_26_cho'}),
                 candles.add_func(ma_lines, col='cho', N=30).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_30_cho', 'EMA': 'ema_30_cho', 'SMA': 'sma_30_cho', 'UB': 'boll_ub_30_cho', 'LB': 'boll_lb_30_cho'}),
                 candles.add_func(ma_lines, col='cho', N=34).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_34_cho', 'EMA': 'ema_34_cho', 'SMA': 'sma_34_cho', 'UB': 'boll_ub_34_cho', 'LB': 'boll_lb_34_cho'}),
                 candles.add_func(ma_lines, col='cho', N=40).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_40_cho', 'EMA': 'ema_40_cho', 'SMA': 'sma_40_cho', 'UB': 'boll_ub_40_cho', 'LB': 'boll_lb_40_cho'}),
                 candles.add_func(ma_lines, col='cho', N=44).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_44_cho', 'EMA': 'ema_44_cho', 'SMA': 'sma_44_cho', 'UB': 'boll_ub_44_cho', 'LB': 'boll_lb_44_cho'}),
                 candles.add_func(ma_lines, col='cho', N=50).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_50_cho', 'EMA': 'ema_50_cho', 'SMA': 'sma_50_cho', 'UB': 'boll_ub_50_cho', 'LB': 'boll_lb_50_cho'}),
                 candles.add_func(ma_lines, col='cho', N=55).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_55_cho', 'EMA': 'ema_55_cho', 'SMA': 'sma_55_cho', 'UB': 'boll_ub_55_cho', 'LB': 'boll_lb_55_cho'}),
                 candles.add_func(ma_lines, col='cho', N=60).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_60_cho', 'EMA': 'ema_60_cho', 'SMA': 'sma_60_cho', 'UB': 'boll_ub_60_cho', 'LB': 'boll_lb_60_cho'}),
                 candles.add_func(ma_lines, col='cho', N=66).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_66_cho', 'EMA': 'ema_66_cho', 'SMA': 'sma_66_cho', 'UB': 'boll_ub_66_cho', 'LB': 'boll_lb_66_cho'}),
                 candles.add_func(ma_lines, col='cho', N=89).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_89_cho', 'EMA': 'ema_89_cho', 'SMA': 'sma_89_cho', 'UB': 'boll_ub_89_cho', 'LB': 'boll_lb_89_cho'}),
                 candles.add_func(ma_lines, col='cho', N=99).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_99_cho', 'EMA': 'ema_99_cho', 'SMA': 'sma_99_cho', 'UB': 'boll_ub_99_cho', 'LB': 'boll_lb_99_cho'}),
                 candles.add_func(ma_lines, col='cho', N=120).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_120_cho', 'EMA': 'ema_120_cho', 'SMA': 'sma_120_cho', 'UB': 'boll_ub_120_cho', 'LB': 'boll_lb_120_cho'}),
                 candles.add_func(ma_lines, col='cho', N=144).loc[:, ['MA', 'EMA', 'SMA', 'UB', 'LB']].rename(columns={'MA': 'ma_144_cho', 'EMA': 'ema_144_cho', 'SMA': 'sma_144_cho', 'UB': 'boll_ub_144_cho', 'LB': 'boll_lb_144_cho'})]

        line = pd.concat(ma + ma1 + ma2 + ma3 + vma + macho + mtm + cci + macd + kdj + skdj + dmi + adtm + vols + pr + asi, axis=1)
        delta = line / (line.shift(1).fillna(0.0) + 0.0000001)
        sec_deriv = delta / (delta.shift(1).fillna(0.0) + 0.0000001)

        delta.columns = ['{}_delta'.format(x) for x in line.columns]
        sec_deriv.columns = ['{}_sec'.format(x) for x in line.columns]

        full = pd.concat([line, delta, sec_deriv, label], axis=1)

        # full = pd.concat([full, full.shift(1).rename(columns=dict([(x, "{}_ld".format(x)) for x in full.columns])), label], axis=1)
        full = full.iloc[150:-10, :].reset_index()
        print("data size: {}".format(full.shape))

        full.loc[full.date < '2018-11-01', :].to_hdf('../data/inds/{}.hdf'.format(stock), key='train')
        full.loc[full.date >= '2018-11-01', :].to_hdf('../data/inds/{}.hdf'.format(stock), key='test')
