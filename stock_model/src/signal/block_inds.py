import pandas as pd
import QUANTAXIS as qa
import datetime as dt


def up_stage(data, col):
    rate = data[col] / (data[col].shift(1)) - 1
    six = (rate > 0.06) * 0.03
    eight = (rate > 0.08) * 0.2
    ten = (rate > 0.097) * 1
    return pd.DataFrame({'hot': six+eight+ten, 'rate': rate})


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    today_str = today.strftime('%Y-%m-%d')

    divide_date = today - dt.timedelta(days=14)
    divide_str = divide_date.strftime('%Y-%m-%d')

    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()
    # stock_list = ZZ800.split('\n')
    blocks = []
    for stock in stock_list:
        print('%s: handling %s' % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stock))
        try:
            block = qa.QA_fetch_stock_block_adv([stock])
        except:
            print('data error during {}'.format(stock))
            continue
        block_names = block.data.reset_index().loc[:, ['code', 'blockname']]
        blocks.append(block_names)
    block_name = pd.concat(blocks, axis=0)
    names = block_name.groupby(['code', 'blockname']).any().reset_index()

    candles = qa.QA_fetch_stock_day_adv(stock_list, start=divide_str, end=today_str).to_qfq()
    stage = candles.add_func(up_stage, 'close').reset_index()
    block_stage = pd.merge(names, stage, on='code')
    hotness = block_stage.groupby(['date', 'blockname'])['hot'].sum()

    hotness['rank'] = hotness.groupby('date')['hot'].rank(ascending=False, method='first')
    hotness['block_hot'] = hotness.apply(lambda x: '{}: {}'.format(x['blockname'], x['hotness']), axis=1)
    hot_pivot = hotness.loc[hotness['rank'] <= 20, :].pivot(index='date', columns='rank', values='block_hot')
    print(hot_pivot)
    hot_pivot.to_csv('../data/block_hotness.csv')

    stock_block = pd.merge(hotness.loc[hotness['rank'] <= 20, :], names, on='blockname')
    top_stock = stock_block.groupby(['date', 'code'])['hotness'].count().reset_index().sort_values(['date', 'hotness'], ascending=[False,False])
    print(top_stock.iloc[:30, :])
    top_stock.to_csv('../data/stock_hot_block.csv')



