import pandas as pd
import QUANTAXIS as qtx
from datetime import datetime, timedelta

TOP_STOCKS = 500
CHECK_DAYS = 30


def mtm_ind(data, col, N=12, M=6):
    mtm = (data[col] - qtx.REF(data[col], N)) / qtx.REF(data[col], N)
    mtmma = qtx.MA(mtm, M)
    return pd.DataFrame({'MTM': mtm, 'MTMMA': mtmma})


if __name__ == '__main__':
    code_list = qtx.QA_fetch_stock_list()
    finfo = qtx.QA_fetch_stock_info(list(code_list.code))
    fcnd = qtx.QA_fetch_stock_day_adv(list(code_list.code),
                                      start=(datetime.today() - timedelta(days=CHECK_DAYS + 30)).strftime('%Y-%m-%d'),
                                      end=datetime.today().strftime('%Y-%m-%d')).to_qfq()
    fbase = fcnd.data.join(finfo.loc[finfo.ipo_date != 0, ['ipo_date', 'industry']], how='inner').reset_index()
    fbase['ipodays'] = fbase.apply(lambda x: (x.date - datetime.strptime(str(x.ipo_date), '%Y%m%d')).days, axis=1)
    fbase = fbase.set_index(['date', 'code'])
    mtm = fcnd.add_func(mtm_ind, col='close', N=20).loc[:, ['MTM', 'MTMMA']]
    smtm = fbase.loc[fbase.ipodays > 180, :].join(mtm)
    smtm['rank'] = smtm.groupby('date')['MTM'].rank(method='dense', ascending=False)

    from quantax import data_query
    raw = data_query.local_get_stock_sw_block(code_list.code.tolist(), '2021-03-15')
    raw['ind_cat'] = raw.apply(lambda x: x.sw1 + '-' + x.sw2, axis=1)
    indict = raw['ind_cat'].to_dict()
    smtm = smtm.reset_index()
    smtm['cat'] = smtm.apply(lambda x: indict[str(x.code)], axis=1)
    smtm = smtm.loc[:, ['date', 'code', 'cat', 'rank']]

    tops = smtm.loc[smtm['rank'] < TOP_STOCKS, :].groupby(['date', 'cat'])['code'].count()
    bases = smtm.groupby(['date', 'cat'])['code'].count()
    rps = pd.merge(tops.reset_index(), bases.reset_index(), on=['date', 'cat'])
    rps.rename(columns={'code_x': 'high', 'code_y': 'full'}, inplace=True)

    rps['score'] = rps.high / rps.full * rps.high
    rps['rank'] = rps.groupby('date')['score'].rank(ascending=False, method='dense', na_option='bottom')
    rps['cap'] = rps.apply(lambda r: '{}/{}'.format(r.high, r.full), axis=1)

    rps = rps.loc[rps.date > (datetime.today() - timedelta(days=CHECK_DAYS)).strftime('%Y-%m-%d'), :]

    ind_pivot = rps.pivot(index='cat', columns='date', values=['score', 'rank', 'cap'])
    ind_pivot = ind_pivot.reindex(sorted(ind_pivot.columns, reverse=True), axis=1)
    ind_pivot.fillna(0, inplace=True)
    ind_pivot.sort_values(by=ind_pivot.columns[0], ascending=False, inplace=True)
    top_ind = ind_pivot.loc[(ind_pivot.score > 1).any(axis=1), :]

    top_ind.to_excel('c:\\Users\\mUSicX\\Desktop\\block_rps_scores{}.xls'.format(TOP_STOCKS))
