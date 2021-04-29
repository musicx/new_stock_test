import pandas as pd
import time
import os
from pymongo import MongoClient

DATE = '2021-03-15'

FSW1 = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\sw_block1.txt'
FSW2 = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\sw_block2.txt'
FSW3 = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\sw_block3.txt'

LCIDX = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\lcidx.lii'
LCLST = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\list.csv'
LCBLK = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\{}.cis'

TDXSW2 = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\ext_block2.txt'
TDXSW3 = 'c:\\Users\\mUSicX\\Documents\\TdxCloud\\Block\\ext_block3.txt'


def get_block(date):
    date_stamp = time.mktime(time.strptime(date, '%Y-%m-%d'))
    try:
        raw_data = pd.DataFrame(
            [
                item for item in coll.find({'date_stamp': date_stamp},
                                           {"_id": 0},
                                           batch_size=10000)
            ]
        )
        ind = raw_data.pivot(index='code', columns='type', values='ind_name')
        ind['sw'] = ind.apply(lambda x: '-'.join([x['sw1'], x['sw2'], x['sw3']]), axis=1)
        ind.reset_index(inplace=True)
        ind['mkt'] = ind.apply(lambda x: 1 if x.code.startswith('6') else 0, axis=1)
        ind.set_index('code', inplace=True)
        return ind
    except Exception as e:
        print(e)
        return None


def first_create():
    lf = open(LCIDX, 'wb')
    ll = open(LCLST, 'w')

    ic = 1
    with open(FSW1, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            line = f.readline()
            line = line.replace('"', '')
            name = line.split(',')[0]
            stocks = line.split(',')[1:]
            len_st = len(stocks)
            if len_st > 0:
                len_up = int(len_st / 256) if len_st > 255 else 0
                len_st = len_st % 256
                ic_full = 393000 + ic
                ic_bytes = (chr(1) + chr(0) * 3 + '{}'.format(ic_full) + chr(0)).encode('ascii') + name[:4].encode('gbk')
                ic_len = len(ic_bytes)
                ic_bytes += bytes(
                    [0] * (52 - ic_len) + [len_st, len_up, 2, 0, 5, 180, 50, 1, 0, 0, 122, 68, 60] + [0] * 255)
                lf.write(ic_bytes)
                ll.write('{},{},{}\n'.format(ic_full, name, len(stocks)))
                ic += 1
                sf = open(LCBLK.format(ic_full), 'wb')
                for s in stocks:
                    ss = chr(1) if s.startswith('6') else chr(0)
                    sf.write((ss + chr(0) + s[:6] + chr(0) * 8).encode('ascii'))
                sf.close()

    ic = 1
    with open(FSW2, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            line = f.readline()
            line = line.replace('"', '')
            name = line.split(',')[0]
            stocks = line.split(',')[1:]
            len_st = len(stocks)
            if len_st > 0:
                len_up = int(len_st / 256) if len_st > 255 else 0
                len_st = len_st % 256
                ic_full = 393100 + ic
                ic_bytes = (chr(1) + chr(0) * 3 + '{}'.format(ic_full) + chr(0)).encode('ascii') + name.split('-')[1][:4].encode('gbk')
                ic_len = len(ic_bytes)
                ic_bytes += bytes(
                    [0] * (52 - ic_len) + [len_st, len_up, 2, 0, 5, 180, 50, 1, 0, 0, 122, 68, 60] + [0] * 255)
                lf.write(ic_bytes)
                ll.write('{},{},{}\n'.format(ic_full, name, len(stocks)))
                ic += 1
                sf = open(LCBLK.format(ic_full), 'wb')
                for s in stocks:
                    ss = chr(1) if s.startswith('6') else chr(0)
                    sf.write((ss + chr(0) + s[:6] + chr(0) * 8).encode('ascii'))
                sf.close()

    ic = 1
    with open(FSW3, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            line = f.readline()
            line = line.replace('"', '')
            name = line.split(',')[0]
            stocks = line.split(',')[1:]
            len_st = len(stocks)
            if len_st > 0:
                len_up = int(len_st / 256) if len_st > 255 else 0
                len_st = len_st % 256
                ic_full = 393300 + ic
                ic_bytes = (chr(1) + chr(0) * 3 + '{}'.format(ic_full) + chr(0)).encode('ascii') + name.split('-')[2][:4].encode('gbk')
                ic_len = len(ic_bytes)
                ic_bytes += bytes(
                    [0] * (52 - ic_len) + [len_st, len_up, 2, 0, 5, 180, 50, 1, 0, 0, 122, 68, 60] + [0] * 255)
                lf.write(ic_bytes)
                ll.write('{},{},{}\n'.format(ic_full, name, len(stocks)))
                ic += 1
                sf = open(LCBLK.format(ic_full), 'wb')
                for s in stocks:
                    ss = chr(1) if s.startswith('6') else chr(0)
                    sf.write((ss + chr(0) + s[:6] + chr(0) * 8).encode('ascii'))
                sf.close()

    lf.close()
    ll.close()


def after_refresh():
    nm = {}
    with open(LCLST, 'r') as f:
        line = f.readline()
        while line:
            nm[line.split(',')[1]] = [line.split(',')[2], line.split(',')[0]]
            line = f.readline()

    lf = open(LCIDX, 'wb')

    with open(FSW1, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            line = f.readline()
            line = line.replace('"', '')
            fullname = line.split(',')[0]
            if len(fullname) > 0:
                name = nm[fullname][0]
                stocks = line.split(',')[1:]
                len_st = len(stocks)
                if len_st > 0:
                    len_up = int(len_st / 256) if len_st > 255 else 0
                    len_st = len_st % 256
                    bc = nm[fullname][1]
                    if len(bc) > 0:
                        ic_bytes = (chr(1) + chr(0) * 3 + '{}'.format(bc) + chr(0)).encode('ascii') + name[:4].encode('gbk')
                        ic_len = len(ic_bytes)
                        ic_bytes += bytes(
                            [0] * (52 - ic_len) + [len_st, len_up, 2, 0, 5, 180, 50, 1, 0, 0, 122, 68, 60] + [0] * 255)
                        lf.write(ic_bytes)
                        sf = open(LCBLK.format(bc), 'wb')
                        for s in stocks:
                            ss = chr(1) if s.startswith('6') else chr(0)
                            sf.write((ss + chr(0) + s[:6] + chr(0) * 8).encode('ascii'))
                        sf.close()

    with open(FSW2, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            line = f.readline()
            line = line.replace('"', '')
            fullname = line.split(',')[0]
            if len(fullname) > 0:
                name = nm[fullname][0]
                stocks = line.split(',')[1:]
                len_st = len(stocks)
                if len_st > 0:
                    len_up = int(len_st / 256) if len_st > 255 else 0
                    len_st = len_st % 256
                    bc = nm[fullname][1]
                    if len(bc) > 0:
                        ic_bytes = (chr(1) + chr(0) * 3 + '{}'.format(bc) + chr(0)).encode('ascii') + name[:4].encode('gbk')
                        ic_len = len(ic_bytes)
                        ic_bytes += bytes(
                            [0] * (52 - ic_len) + [len_st, len_up, 2, 0, 5, 180, 50, 1, 0, 0, 122, 68, 60] + [0] * 255)
                        lf.write(ic_bytes)
                        sf = open(LCBLK.format(bc), 'wb')
                        for s in stocks:
                            ss = chr(1) if s.startswith('6') else chr(0)
                            sf.write((ss + chr(0) + s[:6] + chr(0) * 8).encode('ascii'))
                        sf.close()

    with open(FSW3, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            line = f.readline()
            line = line.replace('"', '')
            fullname = line.split(',')[0]
            if len(fullname) > 0:
                name = nm[fullname][0]
                stocks = line.split(',')[1:]
                len_st = len(stocks)
                if len_st > 0:
                    len_up = int(len_st / 256) if len_st > 255 else 0
                    len_st = len_st % 256
                    bc = nm[fullname][1]
                    if len(bc) > 0:
                        ic_bytes = (chr(1) + chr(0) * 3 + '{}'.format(bc) + chr(0)).encode('ascii') + name[:4].encode('gbk')
                        ic_len = len(ic_bytes)
                        ic_bytes += bytes(
                            [0] * (52 - ic_len) + [len_st, len_up, 2, 0, 5, 180, 50, 1, 0, 0, 122, 68, 60] + [0] * 255)
                        lf.write(ic_bytes)
                        sf = open(LCBLK.format(bc), 'wb')
                        for s in stocks:
                            ss = chr(1) if s.startswith('6') else chr(0)
                            sf.write((ss + chr(0) + s[:6] + chr(0) * 8).encode('ascii'))
                        sf.close()

    lf.close()


if __name__ == '__main__':
    client = MongoClient('mongodb://localhost:27017/')
    db = client['jqdata']
    coll = db['industry']
    block_data = get_block(DATE)
    if block_data is not None:
        tdx2 = block_data.loc[:, ['mkt', 'code', 'sw2']]
        tdx2['v'] = 0.0
        tdx2.to_csv(TDXSW2, sep='|', index=False, header=False, encoding='GB2312')
        tdx3 = block_data.loc[:, ['mkt', 'code', 'sw3']]
        tdx3['v'] = 0.0
        tdx3.to_csv(TDXSW3, sep='|', index=False, header=False, encoding='GB2312')

        s1 = block_data.groupby(['sw1'])['code'].apply(lambda x: ','.join(x))
        s1.to_csv(FSW1, encoding='gbk')
        s2 = block_data.groupby(['sw2'])['code'].apply(lambda x: ','.join(x))
        s2.to_csv(FSW2, encoding='gbk')
        s3 = block_data.groupby(['sw3'])['code'].apply(lambda x: ','.join(x))
        s3.to_csv(FSW3, encoding='gbk')

        if os.path.exists(LCLST):
            after_refresh()
        else:
            first_create()

