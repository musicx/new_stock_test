import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt

today = dt.datetime.today()
start_date = today - dt.timedelta(days=180)
stocks = qa.QA_fetch_stock_list_adv().code.tolist()

cands = qa.QA_fetch_stock_day_adv(stocks, start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))


def lift_stop(data):
    data['lift'] = data.close / data.close.shift(1) - 1
    data['stop'] = (data.lift > 0.095) * 1
    data['uplift'] = ((data.lift > 0.07) & (data.lift <= 0.095)) * 1
    return data.loc[:, ['uplift', 'stop']]

data = cands.add_func(lift_stop)
data = data.loc[:, ['uplift', 'stop']].reset_index()
data = data.sort_values(['code', 'date'], ascending=[True, False])
data['cumstop']= data.groupby('code')['stop'].transform(lambda x: x.cumsum())
data['cumlift']= data.groupby('code')['uplift'].transform(lambda x: x.cumsum())
data['intensity'] = data.cumstop + data.cumlift * 0.5
data['interval'] = (today - data.date).apply(lambda x: x.days)

data['rank'] = data.groupby('interval')['intensity'].transform(lambda x: x.rank(ascending=False))

data.loc[data['rank'] < 100, :].to_excel('../data/intensity.xls')
