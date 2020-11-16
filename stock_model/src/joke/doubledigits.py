import QUANTAXIS as qa
import numpy as np
import pandas as pd
import re, datetime
from joblib import Parallel, delayed

today = datetime.date.today()
begin = today - datetime.timedelta(days=100)


def double_digits(nin):
    ps = '{0:.2f}'.format(nin).replace('.', '')
    r1 = re.search(r'(\d)\1$', ps)
    r2 = re.search(r'(\d{2})\1$', ps)
    # r3 = re.search(r'(\d)\1$', ps)
    return True if r1 or r2 else False


def find(stock):
    candles = qa.QA_fetch_stock_day_adv(stock, start=begin.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    if not candles:
        return None
    da = candles.data
    da['rh'] = da.high.rolling(50).max()
    da['rl'] = da.low.rolling(50).min()
    dp = candles.to_qfq().data
    da['rhr'] = dp.high.rolling(50).max()
    da['rlr'] = dp.low.rolling(50).min()
    mh = da.high.apply(double_digits) & (da.high == da.rh) & (da.rh == da.rhr)
    lh = da.low.apply(double_digits) & (da.low == da.rl) & (da.rl == da.rlr)
    return da.loc[mh | lh, :]


if __name__ == '__main__':
    stocks = qa.QA_fetch_stock_list_adv().code.tolist()
    found = Parallel(n_jobs=4)(delayed(find)(stock) for stock in stocks)

    full = pd.concat([f for f in found if f is not None], axis=0)

    if 'date' in full.columns:
        fs = full.reset_index(drop=True)
    else:
        fs = full.reset_index()
    fs['hh'] = fs.high == fs.rh
    fs['hl'] = fs.low == fs.rl
    earlier = today - datetime.timedelta(days=5)
    fh = fs.loc[fs.date > earlier.strftime('%Y-%m-%d'), ['date', 'code', 'hh', 'hl', 'high', 'low']]

    fh.to_excel('../data/doubledigits.xls')