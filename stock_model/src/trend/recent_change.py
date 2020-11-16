import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from utility import *


if __name__ ==  "__main__":
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    start_date = today - dt.timedelta(days=730)
##    stocks = qa.QA_fetch_stock_list_adv().code.tolist()
    stocks = ['000717']

    for stock in stocks:
        can = qa.QA_fetch_stock_day_adv(stock, start=start_date.strftime('%Y-%m-%d'),
                                        end=today.strftime('%Y-%m-%d')).to_qfq()
        raw = can.data.loc[:, ['open', 'close', 'high', 'low']].values
        dates = [x.strftime('%Y-%m-%d') for x in can.data.reset_index().date.tolist()]
        print('code {}'.format(stock))

        merged, merged_dates, merged_cnt = merge_ochl(raw, dates)
        ends = find_endpoints(merged, merged_cnt)

        cycles = find_cycle(ends, merged_cnt, keep_invalid=False)
        print("found cycles: {}".format(','.join(map(str, cycles))))

        pcycles = find_pair_cycle(cycles)
        print("found paired cycles: {}".format(', '.join(map(str, pcycles))))

        print('-----------')