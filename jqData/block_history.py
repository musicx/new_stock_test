from jqdatasdk import *
from pymongo import MongoClient
import time
import datetime as dt
import pandas as pd

SKIP_LONG = True

if __name__ == '__main__':
    auth('15221518217', 'ZZKww021WW')

    sw_l1 = get_industries(name='sw_l1', date=None)
    sw_l2 = get_industries(name='sw_l2', date=None)
    sw_l3 = get_industries(name='sw_l3', date=None)
    dates = get_trade_days(start_date='2004-02-10')

    client = MongoClient('mongodb://localhost:27017/')
    db = client['jqdata']
    coll = db['industry']
    # try:
    #     coll.create_index([("code", 1), ("date_stamp", 1)], unique=True)
    # except:
    #     pass

    print('start level1 now...')
    # count = 0
    for code, name, time_record in sw_l1.to_records():
        print('doing industry: {}'.format(name))
        # count += 1
        # if count > 5:
        #     break

        start = pd.Timestamp(time_record)
        last_date_cursor = coll.find({'ind_code': code}).sort('date_stamp', -1).limit(1)
        if last_date_cursor.count(with_limit_and_skip=True) > 0:
            last_date = pd.Timestamp(last_date_cursor[0]['date'])
        else:
            last_date = start - pd.Timedelta(days=1)
        print('last date is {}'.format(last_date))
        if (pd.Timestamp(dt.date.today()) - last_date).days > 360 and SKIP_LONG:
            continue
        for date in dates:
            if pd.Timestamp(date) < start or pd.Timestamp(date) <= last_date:
                continue
            stocks = get_industry_stocks(code, date=date.strftime('%Y-%m-%d'))
            date_stamp = time.mktime(time.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d'))
            if len(stocks) > 0:
                print('doing date: {}'.format(date.strftime('%Y-%m-%d')))
                parsed = [{'code': stock[:6], 'type': 'sw1', 'ind_code': code, 'ind_name': name,
                           'date': date.strftime('%Y-%m-%d'), 'date_stamp': date_stamp} for stock in stocks]
                # print(parsed)
                coll.insert_many(parsed)

    # exit(-1)

    print('start level2 now...')
    for code, name, time_record in sw_l2.to_records():
        print('doing industry: {}'.format(name))
        start = pd.Timestamp(time_record)
        last_date_cursor = coll.find({'ind_code': code}).sort('date_stamp', -1).limit(1)
        if last_date_cursor.count(with_limit_and_skip=True) > 0:
            last_date = pd.Timestamp(last_date_cursor[0]['date'])
        else:
            last_date = start - pd.Timedelta(days=1)
        print('last date is {}'.format(last_date))
        if (pd.Timestamp(dt.date.today()) - last_date).days > 360 and SKIP_LONG:
            continue
        for date in dates:
            if pd.Timestamp(date) < start or pd.Timestamp(date) <= last_date:
                continue
            stocks = get_industry_stocks(code, date=date.strftime('%Y-%m-%d'))
            date_stamp = time.mktime(time.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d'))
            if len(stocks) > 0:
                print('doing date: {}'.format(date.strftime('%Y-%m-%d')))
                parsed = [{'code': stock[:6], 'type': 'sw2', 'ind_code': code, 'ind_name': name,
                           'date': date.strftime('%Y-%m-%d'), 'date_stamp': date_stamp} for stock in stocks]
                # print(parsed)
                coll.insert_many(parsed)

    print('start level3 now...')
    for code, name, time_record in sw_l3.to_records():
        print('doing industry: {}'.format(name))
        start = pd.Timestamp(time_record)
        last_date_cursor = coll.find({'ind_code': code}).sort('date_stamp', -1).limit(1)
        if last_date_cursor.count(with_limit_and_skip=True) > 0:
            last_date = pd.Timestamp(last_date_cursor[0]['date'])
        else:
            last_date = start - pd.Timedelta(days=1)
        print('last date is {}'.format(last_date))
        if (pd.Timestamp(dt.date.today()) - last_date).days > 360 and SKIP_LONG:
            continue
        for date in dates:
            if pd.Timestamp(date) < start or pd.Timestamp(date) <= last_date:
                continue
            stocks = get_industry_stocks(code, date=date.strftime('%Y-%m-%d'))
            date_stamp = time.mktime(time.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d'))
            if len(stocks) > 0:
                print('doing date: {}'.format(date.strftime('%Y-%m-%d')))
                parsed = [{'code': stock[:6], 'type': 'sw3', 'ind_code': code, 'ind_name': name,
                           'date': date.strftime('%Y-%m-%d'), 'date_stamp': date_stamp} for stock in stocks]
                # print(parsed)
                coll.insert_many(parsed)
