import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime
from joblib import Parallel, delayed

today = datetime.date.today()
begin = today - datetime.timedelta(days=120)


def prepare_days(stock):
    ca = qa.QA_fetch_stock_day_adv(stock, start=begin.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    if ca is None:
        return None
    data = ca.to_qfq().data
    data['preclose'] = data.close.shift(-1)
    data['s1'] = data.open / data.preclose - 1
    data['s2'] = data.close / data.open - 1
    days = data.loc[:, ['s1', 's2']].stack()
    days.name = stock
    return days.reset_index(level=1, drop=True)


def analyze(data, time):
    corel = data.tail(time * 2).corr()
    corel = corel.stack().reset_index()
    corel = corel.loc[corel.level_0 != corel.level_1, :]
    corel['rk'] = corel.groupby('level_0')[0].transform(pd.DataFrame.rank, ascending=False)
    corel.columns = ['major', 'minor', 'cor', 'rk']
    return corel.loc[(corel.rk <= 5) | (corel.cor > 0.8), :]


if __name__ == '__main__':
    stocks = qa.QA_fetch_stock_list_adv().code.tolist()
    found = Parallel(n_jobs=4)(delayed(prepare_days)(stock) for stock in stocks)
    full = pd.concat([f for f in found if f is not None], axis=1).reset_index(level=1, drop=True).fillna(0)

    corr_90 = analyze(full, 90)
    corr_90.to_excel('../data/correlation.xls')
