# coding :utf-8

import asyncio
import configparser
import datetime
import json
import os
import random
import re
import subprocess
import threading
import time
from multiprocessing import Lock

import numpy as np
import pandas as pd
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient



def QA_util_code_tostr(code):
    """
    explanation:
        将所有沪深股票从数字转化到6位的代码,因为有时候在csv等转换的时候,诸如 000001的股票会变成office强制转化成数字1,
        同时支持聚宽股票格式,掘金股票代码格式,Wind股票代码格式,天软股票代码格式

    params:
        * code ->
            含义: 代码
            类型: str
            参数支持: []
    """
    if isinstance(code, int):
        return "{:>06d}".format(code)
    if isinstance(code, str):
        # 聚宽股票代码格式 '600000.XSHG'
        # 掘金股票代码格式 'SHSE.600000'
        # Wind股票代码格式 '600000.SH'
        # 天软股票代码格式 'SH600000'
        if len(code) == 6:
            return code
        if len(code) == 8:
            # 天软数据
            return code[-6:]
        if len(code) == 9:
            return code[:6]
        if len(code) == 11:
            if code[0] in ["S"]:
                return code.split(".")[1]
            return code.split(".")[0]
        raise ValueError("错误的股票代码格式")
    if isinstance(code, list):
        return QA_util_code_tostr(code[0])


def QA_util_code_tolist(code, auto_fill=True):
    """
    explanation:
        将转换code==> list

    params:
        * code ->
            含义: 代码
            类型: str
            参数支持: []
        * auto_fill->
            含义: 是否自动补全(一般是用于股票/指数/etf等6位数,期货不适用) (default: {True})
            类型: bool
            参数支持: [True]
    """

    if isinstance(code, str):
        if auto_fill:
            return [QA_util_code_tostr(code)]
        else:
            return [code]

    elif isinstance(code, list):
        if auto_fill:
            return [QA_util_code_tostr(item) for item in code]
        else:
            return [item for item in code]


def QA_util_code_adjust_ctp(code, source):
    """
    explanation:
        此函数用于在ctp和通达信之间来回转换

    params:
        * code ->
            含义: 代码
            类型: str
            参数支持: []
        * source->
            含义: 转换至目的源
            类型: str
            参数支持: ["pytdx", "ctp"]

    demonstrate:
        a = QA_util_code_adjust_ctp('AP001', source='ctp')
        b = QA_util_code_adjust_ctp('AP2001', source = 'tdx')
        c = QA_util_code_adjust_ctp('RB2001', source = 'tdx')
        d =  QA_util_code_adjust_ctp('rb2001', source = 'ctp')
        print(a+"\n"+b+"\n"+c+"\n"+d)

    output:
        >>AP2001
        >>AP001
        >>rb2001
        >>RB2001
    """
    if source == 'ctp':
        if len(re.search(r'[0-9]+', code)[0]) < 4:
            return re.search(r'[a-zA-z]+', code)[0] + '2' + re.search(r'[0-9]+', code)[0]
        else:
            return code.upper()
    else:
        if re.search(r'[a-zA-z]+', code)[0].upper() in ['RM', 'CJ', 'OI', 'CY', 'AP', 'SF', 'SA', 'UR', 'FG', 'LR',
                                                        'CF', 'WH', 'IPS', 'ZC', 'SPD', 'MA', 'TA', 'JR', 'SM', 'PM',
                                                        'RS', 'SR', 'RI']:
            return re.search(r'[a-zA-z]+', code)[0] + re.search(r'[0-9]+', code)[0][1:]
        else:
            return re.search(r'[a-zA-z]+', code)[0].lower() + re.search(r'[0-9]+', code)[0]


QATZInfo_CN = 'Asia/Shanghai'


def QA_util_time_now():
    """
    explanation:
       获取当前日期时间

    return:
        datetime
    """
    return datetime.datetime.now()


def QA_util_date_today():
    """
    explanation:
       获取当前日期

    return:
        date
    """
    return datetime.date.today()


def QA_util_today_str():
    """
    explanation:
        返回今天的日期字符串

    return:
        str
    """
    dt = QA_util_date_today()
    return QA_util_datetime_to_strdate(dt)


def QA_util_date_str2int(date):
    """
    explanation:
        转换日期字符串为整数

    params:
        * date->
            含义: 日期字符串
            类型: date
            参数支持: []

    demonstrate:
        print(QA_util_date_str2int("2011-09-11"))

    return:
        int

    output:
        >>20110911
    """
    # return int(str(date)[0:4] + str(date)[5:7] + str(date)[8:10])
    if isinstance(date, str):
        return int(str().join(date.split('-')))
    elif isinstance(date, int):
        return date


def QA_util_date_int2str(int_date):
    """
    explanation:
        转换日期整数为字符串

    params:
        * int_date->
            含义: 日期转换得
            类型: int
            参数支持: []

    return:
        str
    """
    date = str(int_date)
    if len(date) == 8:
        return str(date[0:4] + '-' + date[4:6] + '-' + date[6:8])
    elif len(date) == 10:
        return date


def QA_util_to_datetime(time):
    """
    explanation:
        转换字符串格式的日期为datetime

    params:
        * time->
            含义: 日期
            类型: str
            参数支持: []

    return:
        datetime
    """
    if len(str(time)) == 10:
        _time = '{} 00:00:00'.format(time)
    elif len(str(time)) == 19:
        _time = str(time)
    else:
        print('WRONG DATETIME FORMAT {}'.format(time))
    return datetime.datetime.strptime(_time, '%Y-%m-%d %H:%M:%S')


def QA_util_datetime_to_strdate(dt):
    """
    explanation:
        转换字符串格式的日期为datetime

    params:
        * dt->
            含义: 日期时间
            类型: datetime
            参数支持: []

    return:
        str
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate


def QA_util_datetime_to_strdatetime(dt):
    """
    explanation:
        转换日期时间为字符串格式

    params:
        * dt->
            含义: 日期时间
            类型: datetime
            参数支持: []

    return:
        datetime

    """
    strdatetime = "%04d-%02d-%02d %02d:%02d:%02d" % (
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second
    )
    return strdatetime


def QA_util_date_stamp(date):
    """
    explanation:
        转换日期时间字符串为浮点数的时间戳

    params:
        * date->
            含义: 日期时间
            类型: str
            参数支持: []

    return:
        time
    """
    datestr = pd.Timestamp(date).strftime("%Y-%m-%d")
    date = time.mktime(time.strptime(datestr, '%Y-%m-%d'))
    return date


def QA_util_time_stamp(time_):
    """
    explanation:
       转换日期时间的字符串为浮点数的时间戳

    params:
        * time_->
            含义: 日期时间
            类型: str
            参数支持: ['2018-01-01 00:00:00']

    return:
        time
    """
    if len(str(time_)) == 10:
        # yyyy-mm-dd格式
        return time.mktime(time.strptime(time_, '%Y-%m-%d'))
    elif len(str(time_)) == 16:
        # yyyy-mm-dd hh:mm格式
        return time.mktime(time.strptime(time_, '%Y-%m-%d %H:%M'))
    else:
        timestr = str(time_)[0:19]
        return time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))


def QA_util_tdxtimestamp(time_stamp):
    """
    explanation:
       转换tdx的realtimeQuote数据, [相关地址](https://github.com/rainx/pytdx/issues/187#issuecomment-441270487)

    params:
        * time_stamp->
            含义: 时间
            类型: str
            参数支持: []

    return:
        int

    """
    if time_stamp is not None:
        time_stamp = str(time_stamp)
        time = time_stamp[:-6] + ':'
        if int(time_stamp[-6:-4]) < 60:
            time += '%s:' % time_stamp[-6:-4]
            time += '%06.3f' % (
                    int(time_stamp[-4:]) * 60 / 10000.0
            )
        else:
            time += '%02d:' % (
                    int(time_stamp[-6:]) * 60 / 1000000
            )
            time += '%06.3f' % (
                    (int(time_stamp[-6:]) * 60 % 1000000) * 60 / 1000000.0
            )
        return time


def QA_util_pands_timestamp_to_date(pandsTimestamp):
    """
    explanation:
        转换 pandas 的时间戳 到 datetime.date类型

    params:
        * pandsTimestamp->
            含义: pandas的时间戳
            类型:  pandas._libs.tslib.Timestamp
            参数支持: []
    return:
        date
    """
    return pandsTimestamp.to_pydatetime().date()


def QA_util_pands_timestamp_to_datetime(pandsTimestamp):
    """
    explanation:
        转换 pandas时间戳 到 datetime.datetime类型

    params:
        * pandsTimestamp->
            含义: pandas时间戳
            类型:  pandas._libs.tslib.Timestamp
            参数支持: []
    return:
        datetime
    """
    return pandsTimestamp.to_pydatetime()


def QA_util_stamp2datetime(timestamp):
    """
    explanation:
        datestamp转datetime,pandas转出来的timestamp是13位整数 要/1000,
        It’s common for this to be restricted to years from 1970 through 2038.
        从1970年开始的纳秒到当前的计数 转变成 float 类型时间 类似 time.time() 返回的类型

    params:
        * timestamp->
            含义: 时间戳
            类型: float
            参数支持: []

    return:
        datetime
    """
    try:
        return datetime.datetime.fromtimestamp(timestamp)
    except Exception as e:
        # it won't work ??
        try:
            return datetime.datetime.fromtimestamp(timestamp / 1000)
        except:
            try:
                return datetime.datetime.fromtimestamp(timestamp / 1000000)
            except:
                return datetime.datetime.fromtimestamp(timestamp / 1000000000)

    #


def QA_util_ms_stamp(ms):
    """
    explanation:
        直接返回不做处理

    params:
        * ms->
            含义: 时间戳
            类型: float
            参数支持: []
    return:
        float
    """

    return ms


def QA_util_date_valid(date):
    """
    explanation:
        判断字符串格式(1982-05-11)

    params:
        * date->
            含义: 日期
            类型: str
            参数支持: []

    return:
        bool
    """
    try:
        time.strptime(date, "%Y-%m-%d")
        return True
    except:
        return False


def QA_util_realtime(strtime, client):
    """
    explanation:
        查询数据库中的数据

    params:
        * strtime->
            含义: 日期
            类型: str
            参数支持: []
        * client->
            含义: 源
            类型: pymongo.MongoClient
            参数支持: []

    return:
        dict
    """
    time_stamp = QA_util_date_stamp(strtime)
    coll = client.quantaxis.trade_date
    temp_str = coll.find_one({'date_stamp': {"$gte": time_stamp}})
    time_real = temp_str['date']
    time_id = temp_str['num']
    return {'time_real': time_real, 'id': time_id}


def QA_util_id2date(idx, client):
    """
    explanation:
         从数据库中查询通达信时间

    params:
        * idx->
            含义: 数据库index
            类型: str
            参数支持: []
        * client->
            含义: 源
            类型: pymongo.MongoClient
            参数支持: []

    return:
        str
    """
    coll = client.quantaxis.trade_date
    temp_str = coll.find_one({'num': idx})
    return temp_str['date']


def QA_util_is_trade(date, code, client):
    """
    explanation:
        从数据库中查询判断是否是交易日

    params:
        * date->
            含义: 日期
            类型: str
            参数支持: []
        * code->
            含义: 代码
            类型: str
            参数支持: []
        * client->
            含义: 源
            类型: pymongo.MongoClient
            参数支持: []

    return:
        bool
    """
    coll = client.quantaxis.stock_day
    date = str(date)[0:10]
    is_trade = coll.find_one({'code': code, 'date': date})
    try:
        len(is_trade)
        return True
    except:
        return False


def QA_util_get_date_index(date, trade_list):
    """
    explanation:
        返回在trade_list中的index位置

    params:
        * date->
            含义: 日期
            类型: str
            参数支持: []
        * trade_list->
            含义: 代码
            类型: ??
            参数支持: []

    return:
        ??
    """
    return trade_list.index(date)


def QA_util_get_index_date(id, trade_list):
    """
    explanation:
        根据id索引值

    params:
        * id->
            含义: 日期
            类型: str
            参数支持: []
        * trade_list->
            含义: 代码
            类型: dict
            参数支持: []

    return:
        ??
    """
    return trade_list[id]


def QA_util_select_hours(time=None, gt=None, lt=None, gte=None, lte=None):
    """
    explanation:
        quantaxis的时间选择函数,约定时间的范围,比如早上9点到11点

    params:
        * time->
            含义: 时间
            类型: str
            参数支持: []
        * gt->
            含义: 大于
            类型: Any
            参数支持: []
        * lt->
            含义: 小于
            类型: Any
            参数支持: []
        * gte->
            含义: 大于等于
            类型: Any
            参数支持: []
        * lte->
            含义: 小于等于
            类型: Any
            参数支持: []

    return:
        bool
    """
    if time is None:
        __realtime = datetime.datetime.now()
    else:
        __realtime = time

    fun_list = []
    if gt != None:
        fun_list.append('>')
    if lt != None:
        fun_list.append('<')
    if gte != None:
        fun_list.append('>=')
    if lte != None:
        fun_list.append('<=')

    assert len(fun_list) > 0
    true_list = []
    try:
        for item in fun_list:
            if item == '>':
                if __realtime.strftime('%H') > gt:
                    true_list.append(0)
                else:
                    true_list.append(1)
            elif item == '<':
                if __realtime.strftime('%H') < lt:
                    true_list.append(0)
                else:
                    true_list.append(1)
            elif item == '>=':
                if __realtime.strftime('%H') >= gte:
                    true_list.append(0)
                else:
                    true_list.append(1)
            elif item == '<=':
                if __realtime.strftime('%H') <= lte:
                    true_list.append(0)
                else:
                    true_list.append(1)

    except:
        return Exception
    if sum(true_list) > 0:
        return False
    else:
        return True


def QA_util_select_min(time=None, gt=None, lt=None, gte=None, lte=None):
    """
    explanation:
        择分钟

    params:
        * time->
            含义: 时间
            类型: str
            参数支持: []
        * gt->
            含义: 大于等于
            类型: Any
            参数支持: []
        * lt->
            含义: 小于
            类型: Any
            参数支持: []
        * gte->
            含义: 大于等于
            类型: Any
            参数支持: []
        * lte->
            含义: 小于等于
            类型: Any
            参数支持: []

    return:
        bool
    """
    if time is None:
        __realtime = datetime.datetime.now()
    else:
        __realtime = time

    fun_list = []
    if gt != None:
        fun_list.append('>')
    if lt != None:
        fun_list.append('<')
    if gte != None:
        fun_list.append('>=')
    if lte != None:
        fun_list.append('<=')

    assert len(fun_list) > 0
    true_list = []
    try:
        for item in fun_list:
            if item == '>':
                if __realtime.strftime('%M') > gt:
                    true_list.append(0)
                else:
                    true_list.append(1)
            elif item == '<':
                if __realtime.strftime('%M') < lt:
                    true_list.append(0)
                else:
                    true_list.append(1)
            elif item == '>=':
                if __realtime.strftime('%M') >= gte:
                    true_list.append(0)
                else:
                    true_list.append(1)
            elif item == '<=':
                if __realtime.strftime('%M') <= lte:
                    true_list.append(0)
                else:
                    true_list.append(1)
    except:
        return Exception
    if sum(true_list) > 0:
        return False
    else:
        return True


def QA_util_time_delay(time_=0):
    """
    explanation:
        这是一个用于复用/比如说@装饰器的延时函数,使用threading里面的延时,为了是不阻塞进程,
        有时候,同时发进去两个函数,第一个函数需要延时,第二个不需要的话,用sleep就会阻塞掉第二个进程

    params:
        * time_->
            含义: 时间
            类型: time
            参数支持: []

    return:
        func
    """

    def _exec(func):
        threading.Timer(time_, func)

    return _exec


def QA_util_calc_time(func, *args, **kwargs):
    """
    explanation:
        耗时长度的装饰器

    params:
        * func ->
            含义: 被装饰的函数
            类型: func
            参数支持: []
        * args ->
            含义: 函数接受的任意元组参数
            类型: tuple
            参数支持: []
        * kwargs ->
            含义: 函数接受的任意字典参数
            类型: dict
            参数支持: []

    return:
        None
    """
    _time = datetime.datetime.now()
    func(*args, **kwargs)
    print(datetime.datetime.now() - _time)
    # return datetime.datetime.now() - _time


def QA_util_to_json_from_pandas(data):
    """
    explanation:
        将pandas数据转换成json格式

    params:
        * data ->:
            meaning: pandas数据
            type: null
            optional: [null]

    return:
        dict

    demonstrate:
        Not described

    output:
        Not described
    """

    """需要对于datetime 和date 进行转换, 以免直接被变成了时间戳"""
    if 'datetime' in data.columns:
        data.datetime = data.datetime.apply(str)
    if 'date' in data.columns:
        data.date = data.date.apply(str)
    return json.loads(data.to_json(orient='records'))


def QA_util_to_list_from_pandas(data):
    """
    explanation:
         将pandas数据转换成列表

    params:
        * data ->:
            meaning: pandas数据
            type: null
            optional: [null]

    return:
        list

    demonstrate:
        Not described

    output:
        Not described
    """

    return np.asarray(data).tolist()


def QA_util_to_list_from_numpy(data):
    """
    explanation:
        将numpy数据转换为列表

    params:
        * data ->:
            meaning: numpy数据
            type: null
            optional: [null]

    return:
        None

    demonstrate:
        Not described

    output:
        Not described
    """

    return data.tolist()


def QA_util_to_pandas_from_json(data):
    """
    explanation:
        将json数据载入为pandas数据

    params:
        * data ->:
            meaning: json数据
            type: null
            optional: [null]

    return:
        DataFrame

    demonstrate:
        Not described

    output:
        Not described
    """
    if isinstance(data, dict):
        return pd.DataFrame(data=[data, ])
    else:
        return pd.DataFrame(data=[{'value': data}])


def QA_util_to_pandas_from_list(data):
    """
    explanation:
        将列表数据转换为pandas

    params:
        * data ->:
            meaning: 列表数据
            type: list
            optional: [null]

    return:
        DataFrame

    demonstrate:
        Not described

    output:
        Not described
    """

    if isinstance(data, list):
        return pd.DataFrame(data=data)

#
# def QA_util_mongo_initial(db=DATABASE):
#     db.drop_collection('stock_day')
#     db.drop_collection('stock_list')
#     db.drop_collection('stock_info')
#     db.drop_collection('trade_date')
#     db.drop_collection('stock_min')
#     db.drop_collection('stock_transaction')
#     db.drop_collection('stock_xdxr')
#
#
# def QA_util_mongo_status(db=DATABASE):
#     print(db.collection_names())
#     print(db.last_status())
#     print(subprocess.call('mongostat', shell=True))
#
#
# def QA_util_mongo_infos(db=DATABASE):
#     data_struct = []
#
#     for item in db.collection_names():
#         value = []
#         value.append(item)
#         value.append(eval('db.' + str(item) + '.find({}).count()'))
#         value.append(list(eval('db.' + str(item) + '.find_one()').keys()))
#         data_struct.append(value)
#     return pd.DataFrame(data_struct, columns=['collection_name', 'counts', 'columns']).set_index('collection_name')


ASCENDING = pymongo.ASCENDING
DESCENDING = pymongo.DESCENDING
QA_util_sql_mongo_sort_ASCENDING = pymongo.ASCENDING
QA_util_sql_mongo_sort_DESCENDING = pymongo.DESCENDING



def exclude_from_stock_ip_list(exclude_ip_list):
    """
    explanation:
        从stock_ip_list删除列表exclude_ip_list中的ip,从stock_ip_list删除列表future_ip_list中的ip

    params:
        * exclude_ip_list ->:
            meaning: 需要删除的ip_list
            type: list
            optional: [null]

    return:
        None

    demonstrate:
        Not described

    output:
        Not described
    """
    for exc in exclude_ip_list:
        if exc in stock_ip_list:
            stock_ip_list.remove(exc)

    # 扩展市场
    for exc in exclude_ip_list:
        if exc in future_ip_list:
            future_ip_list.remove(exc)



def QA_util_random_with_zh_stock_code(stockNumber=10):
    """
    explanation:
        随机生成股票代码

    params:
        * stockNumber ->:
            meaning: 生成个数
            type: int
            optional: [10]

    return:
        List

    demonstrate:
        Not described

    output:
        Not described
    """
    codeList = []
    pt = 0
    for i in range(stockNumber):
        if pt == 0:
            # print("random 60XXXX")
            iCode = random.randint(600000, 609999)
            aCode = "%06d" % iCode

        elif pt == 1:
            # print("random 00XXXX")
            iCode = random.randint(600000, 600999)
            aCode = "%06d" % iCode

        elif pt == 2:
            # print("random 00XXXX")
            iCode = random.randint(2000, 9999)
            aCode = "%06d" % iCode

        elif pt == 3:
            # print("random 300XXX")
            iCode = random.randint(300000, 300999)
            aCode = "%06d" % iCode

        else:
            # print("random 00XXXX")
            iCode = random.randint(2000, 2999)
            aCode = "%06d" % iCode
        pt = (pt + 1) % 5
        codeList.append(aCode)
    return codeList


def QA_util_random_with_topic(topic='Acc', lens=8):
    """
    explanation:
        生成account随机值

    params:
        * stockNutopicmber ->:
            meaning: 开头
            type: str
            optional: ['Acc']
        * lens ->:
            meaning: 长度
            type: int
            optional: [10]

    return:
        str

    demonstrate:
        Not described

    output:
        Not described
    """

    _list = [chr(i) for i in range(65,
                                   91)] + [chr(i) for i in range(97,
                                                                 123)
                                           ] + [str(i) for i in range(10)]

    num = random.sample(_list, lens)
    return '{}_{}'.format(topic, ''.join(num))


month_data = pd.date_range(
    '1/1/1996',
    '12/31/2023',
    freq='Q-MAR'
).astype(str).tolist()

financial_dict = {

    # 1.每股指标
    '001基本每股收益': 'EPS',
    '002扣除非经常性损益每股收益': 'deductEPS',
    '003每股未分配利润': 'undistributedProfitPerShare',
    '004每股净资产': 'netAssetsPerShare',
    '005每股资本公积金': 'capitalReservePerShare',
    '006净资产收益率': 'ROE',
    '007每股经营现金流量': 'operatingCashFlowPerShare',
    # 2. 资产负债表 BALANCE SHEET
    # 2.1 资产
    # 2.1.1 流动资产
    '008货币资金': 'moneyFunds',
    '009交易性金融资产': 'tradingFinancialAssets',
    '010应收票据': 'billsReceivables',
    '011应收账款': 'accountsReceivables',
    '012预付款项': 'prepayments',
    '013其他应收款': 'otherReceivables',
    '014应收关联公司款': 'interCompanyReceivables',
    '015应收利息': 'interestReceivables',
    '016应收股利': 'dividendsReceivables',
    '017存货': 'inventory',
    '018其中：消耗性生物资产': 'expendableBiologicalAssets',
    '019一年内到期的非流动资产': 'noncurrentAssetsDueWithinOneYear',
    '020其他流动资产': 'otherLiquidAssets',
    '021流动资产合计': 'totalLiquidAssets',
    # 2.1.2 非流动资产
    '022可供出售金融资产': 'availableForSaleSecurities',
    '023持有至到期投资': 'heldToMaturityInvestments',
    '024长期应收款': 'longTermReceivables',
    '025长期股权投资': 'longTermEquityInvestment',
    '026投资性房地产': 'investmentRealEstate',
    '027固定资产': 'fixedAssets',
    '028在建工程': 'constructionInProgress',
    '029工程物资': 'engineerMaterial',
    '030固定资产清理': 'fixedAssetsCleanUp',
    '031生产性生物资产': 'productiveBiologicalAssets',
    '032油气资产': 'oilAndGasAssets',
    '033无形资产': 'intangibleAssets',
    '034开发支出': 'developmentExpenditure',
    '035商誉': 'goodwill',
    '036长期待摊费用': 'longTermDeferredExpenses',
    '037递延所得税资产': 'deferredIncomeTaxAssets',
    '038其他非流动资产': 'otherNonCurrentAssets',
    '039非流动资产合计': 'totalNonCurrentAssets',
    '040资产总计': 'totalAssets',
    # 2.2 负债
    # 2.2.1 流动负债
    '041短期借款': 'shortTermLoan',
    '042交易性金融负债': 'tradingFinancialLiabilities',
    '043应付票据': 'billsPayable',
    '044应付账款': 'accountsPayable',
    '045预收款项': 'advancedReceivable',
    '046应付职工薪酬': 'employeesPayable',
    '047应交税费': 'taxPayable',
    '048应付利息': 'interestPayable',
    '049应付股利': 'dividendPayable',
    '050其他应付款': 'otherPayable',
    '051应付关联公司款': 'interCompanyPayable',
    '052一年内到期的非流动负债': 'noncurrentLiabilitiesDueWithinOneYear',
    '053其他流动负债': 'otherCurrentLiabilities',
    '054流动负债合计': 'totalCurrentLiabilities',
    # 2.2.2 非流动负债
    '055长期借款': 'longTermLoans',
    '056应付债券': 'bondsPayable',
    '057长期应付款': 'longTermPayable',
    '058专项应付款': 'specialPayable',
    '059预计负债': 'estimatedLiabilities',
    '060递延所得税负债': 'defferredIncomeTaxLiabilities',
    '061其他非流动负债': 'otherNonCurrentLiabilities',
    '062非流动负债合计': 'totalNonCurrentLiabilities',
    '063负债合计': 'totalLiabilities',
    # 2.3 所有者权益
    '064实收资本（或股本）': 'totalShare',
    '065资本公积': 'capitalReserve',
    '066盈余公积': 'surplusReserve',
    '067减：库存股': 'treasuryStock',
    '068未分配利润': 'undistributedProfits',
    '069少数股东权益': 'minorityEquity',
    '070外币报表折算价差': 'foreignCurrencyReportTranslationSpread',
    '071非正常经营项目收益调整': 'abnormalBusinessProjectEarningsAdjustment',
    '072所有者权益（或股东权益）合计': 'totalOwnersEquity',
    '073负债和所有者（或股东权益）合计': 'totalLiabilitiesAndOwnersEquity',
    # 3. 利润表
    '074其中：营业收入': 'operatingRevenue',
    '075其中：营业成本': 'operatingCosts',
    '076营业税金及附加': 'taxAndSurcharges',
    '077销售费用': 'salesCosts',
    '078管理费用': 'managementCosts',
    '079堪探费用': 'explorationCosts',
    '080财务费用': 'financialCosts',
    '081资产减值损失': 'assestsDevaluation',
    '082加：公允价值变动净收益': 'profitAndLossFromFairValueChanges',
    '083投资收益': 'investmentIncome',
    '084其中：对联营企业和合营企业的投资收益': 'investmentIncomeFromAffiliatedBusinessAndCooperativeEnterprise',
    '085影响营业利润的其他科目': 'otherSubjectsAffectingOperatingProfit',
    '086三、营业利润': 'operatingProfit',
    '087加：补贴收入': 'subsidyIncome',
    '088营业外收入': 'nonOperatingIncome',
    '089减：营业外支出': 'nonOperatingExpenses',
    '090其中：非流动资产处置净损失': 'netLossFromDisposalOfNonCurrentAssets',
    '091加：影响利润总额的其他科目': 'otherSubjectsAffectTotalProfit',
    '092四、利润总额': 'totalProfit',
    '093减：所得税': 'incomeTax',
    '094加：影响净利润的其他科目': 'otherSubjectsAffectNetProfit',
    '095五、净利润': 'netProfit',
    '096归属于母公司所有者的净利润': 'netProfitsBelongToParentCompanyOwner',
    '097少数股东损益': 'minorityProfitAndLoss',

    # 4. 现金流量表
    # 4.1 经营活动 Operating
    '098销售商品、提供劳务收到的现金': 'cashFromGoodsSalesorOrRenderingOfServices',
    '099收到的税费返还': 'refundOfTaxAndFeeReceived',
    '100收到其他与经营活动有关的现金': 'otherCashRelatedBusinessActivitiesReceived',
    '101经营活动现金流入小计': 'cashInflowsFromOperatingActivities',
    '102购买商品、接受劳务支付的现金': 'buyingGoodsReceivingCashPaidForLabor',
    '103支付给职工以及为职工支付的现金': 'paymentToEmployeesAndCashPaidForEmployees',
    '104支付的各项税费': 'paymentsOfVariousTaxes',
    '105支付其他与经营活动有关的现金': 'paymentOfOtherCashRelatedToBusinessActivities',
    '106经营活动现金流出小计': 'cashOutflowsFromOperatingActivities',
    '107经营活动产生的现金流量净额': 'netCashFlowsFromOperatingActivities',
    # 4.2 投资活动 Investment
    '108收回投资收到的现金': 'cashReceivedFromInvestmentReceived',
    '109取得投资收益收到的现金': 'cashReceivedFromInvestmentIncome',
    '110处置固定资产、无形资产和其他长期资产收回的现金净额': 'disposalOfNetCashForRecoveryOfFixedAssetsIntangibleAssetsAndOtherLongTermAssets',
    '111处置子公司及其他营业单位收到的现金净额': 'disposalOfNetCashReceivedFromSubsidiariesAndOtherBusinessUnits',
    '112收到其他与投资活动有关的现金': 'otherCashReceivedRelatingToInvestingActivities',
    '113投资活动现金流入小计': 'cashinFlowsFromInvestmentActivities',
    '114购建固定资产、无形资产和其他长期资产支付的现金': 'cashForThePurchaseConstructionPaymentOfFixedAssetsIntangibleAssetsAndOtherLongTermAssets',
    '115投资支付的现金': 'cashInvestment',
    '116取得子公司及其他营业单位支付的现金净额': 'acquisitionOfNetCashPaidBySubsidiariesAndOtherBusinessUnits',
    '117支付其他与投资活动有关的现金': 'otherCashPaidRelatingToInvestingActivities',
    '118投资活动现金流出小计': 'cashOutflowsFromInvestmentActivities',
    '119投资活动产生的现金流量净额': 'netCashFlowsFromInvestingActivities',
    # 4.3 筹资活动 Financing
    '120吸收投资收到的现金': 'cashReceivedFromInvestors',
    '121取得借款收到的现金': 'cashFromBorrowings',
    '122收到其他与筹资活动有关的现金': 'otherCashReceivedRelatingToFinancingActivities',
    '123筹资活动现金流入小计': 'cashInflowsFromFinancingActivities',
    '124偿还债务支付的现金': 'cashPaymentsOfAmountBorrowed',
    '125分配股利、利润或偿付利息支付的现金': 'cashPaymentsForDistrbutionOfDividendsOrProfits',
    '126支付其他与筹资活动有关的现金': 'otherCashPaymentRelatingToFinancingActivities',
    '127筹资活动现金流出小计': 'cashOutflowsFromFinancingActivities',
    '128筹资活动产生的现金流量净额': 'netCashFlowsFromFinancingActivities',
    # 4.4 汇率变动
    '129四、汇率变动对现金的影响': 'effectOfForeignExchangRateChangesOnCash',
    '130四(2)、其他原因对现金的影响': 'effectOfOtherReasonOnCash',
    # 4.5 现金及现金等价物净增加
    '131五、现金及现金等价物净增加额': 'netIncreaseInCashAndCashEquivalents',
    '132期初现金及现金等价物余额': 'initialCashAndCashEquivalentsBalance',
    # 4.6 期末现金及现金等价物余额
    '133期末现金及现金等价物余额': 'theFinalCashAndCashEquivalentsBalance',
    # 4.x 补充项目 Supplementary Schedule：
    # 现金流量附表项目    Indirect Method
    # 4.x.1 将净利润调节为经营活动现金流量 Convert net profit to cash flow from operating activities
    '134净利润': 'netProfitFromOperatingActivities',
    '135资产减值准备': 'provisionForAssetsLosses',
    '136固定资产折旧、油气资产折耗、生产性生物资产折旧': 'depreciationForFixedAssets',
    '137无形资产摊销': 'amortizationOfIntangibleAssets',
    '138长期待摊费用摊销': 'amortizationOfLong-termDeferredExpenses',
    '139处置固定资产、无形资产和其他长期资产的损失': 'lossOfDisposingFixedAssetsIntangibleAssetsAndOtherLong-termAssets',
    '140固定资产报废损失': 'scrapLossOfFixedAssets',
    '141公允价值变动损失': 'lossFromFairValueChange',
    '142财务费用': 'financialExpenses',
    '143投资损失': 'investmentLosses',
    '144递延所得税资产减少': 'decreaseOfDeferredTaxAssets',
    '145递延所得税负债增加': 'increaseOfDeferredTaxLiabilities',
    '146存货的减少': 'decreaseOfInventory',
    '147经营性应收项目的减少': 'decreaseOfOperationReceivables',
    '148经营性应付项目的增加': 'increaseOfOperationPayables',
    '149其他': 'others',
    '150经营活动产生的现金流量净额2': 'netCashFromOperatingActivities2',
    # 4.x.2 不涉及现金收支的投资和筹资活动 Investing and financing activities not involved in cash
    '151债务转为资本': 'debtConvertedToCSapital',
    '152一年内到期的可转换公司债券': 'convertibleBondMaturityWithinOneYear',
    '153融资租入固定资产': 'leaseholdImprovements',
    # 4.x.3 现金及现金等价物净增加情况 Net increase of cash and cash equivalents
    '154现金的期末余额': 'cashEndingBal',
    '155现金的期初余额': 'cashBeginingBal',
    '156现金等价物的期末余额': 'cashEquivalentsEndingBal',
    '157现金等价物的期初余额': 'cashEquivalentsBeginningBal',
    '158现金及现金等价物净增加额': 'netIncreaseOfCashAndCashEquivalents',
    # 5. 偿债能力分析
    '159流动比率': 'currentRatio',  # 流动资产/流动负债
    '160速动比率': 'acidTestRatio',  # (流动资产-存货）/流动负债
    '161现金比率(%)': 'cashRatio',  # (货币资金+有价证券)÷流动负债
    '162利息保障倍数': 'interestCoverageRatio',  # (利润总额+财务费用（仅指利息费用部份）)/利息费用
    '163非流动负债比率(%)': 'noncurrentLiabilitiesRatio',
    '164流动负债比率(%)': 'currentLiabilitiesRatio',
    '165现金到期债务比率(%)': 'cashDebtRatio',  # 企业经营现金净流入/(本期到期长期负债+本期应付票据)
    '166有形资产净值债务率(%)': 'debtToTangibleAssetsRatio',
    '167权益乘数(%)': 'equityMultiplier',  # 资产总额/股东权益总额
    '168股东的权益/负债合计(%)': 'equityDebtRatio',  # 权益负债率
    '169有形资产/负债合计(%)': 'tangibleAssetDebtRatio ',  # 有形资产负债率
    '170经营活动产生的现金流量净额/负债合计(%)': 'netCashFlowsFromOperatingActivitiesDebtRatio',
    '171EBITDA/负债合计(%)': 'EBITDA/Liabilities',
    # 6. 经营效率分析
    # 销售收入÷平均应收账款=销售收入\(0.5 x(应收账款期初+期末))
    '172应收帐款周转率': 'turnoverRatioOfReceivable;',
    '173存货周转率': 'turnoverRatioOfInventory',
    # (存货周转天数+应收帐款周转天数-应付帐款周转天数+预付帐款周转天数-预收帐款周转天数)/365
    '174运营资金周转率': 'turnoverRatioOfOperatingAssets',
    '175总资产周转率': 'turnoverRatioOfTotalAssets',
    '176固定资产周转率': 'turnoverRatioOfFixedAssets',  # 企业销售收入与固定资产净值的比率
    '177应收帐款周转天数': 'daysSalesOutstanding',  # 企业从取得应收账款的权利到收回款项、转换为现金所需要的时间
    '178存货周转天数': 'daysSalesOfInventory',  # 企业从取得存货开始，至消耗、销售为止所经历的天数
    '179流动资产周转率': 'turnoverRatioOfCurrentAssets',  # 流动资产周转率(次)=主营业务收入/平均流动资产总额
    '180流动资产周转天数': 'daysSalesofCurrentAssets',
    '181总资产周转天数': 'daysSalesofTotalAssets',
    '182股东权益周转率': 'equityTurnover',  # 销售收入/平均股东权益
    # 7. 发展能力分析
    '183营业收入增长率(%)': 'operatingIncomeGrowth',
    '184净利润增长率(%)': 'netProfitGrowthRate',  # NPGR  利润总额－所得税
    '185净资产增长率(%)': 'netAssetsGrowthRate',
    '186固定资产增长率(%)': 'fixedAssetsGrowthRate',
    '187总资产增长率(%)': 'totalAssetsGrowthRate',
    '188投资收益增长率(%)': 'investmentIncomeGrowthRate',
    '189营业利润增长率(%)': 'operatingProfitGrowthRate',
    '190暂无': 'None1',
    '191暂无': 'None2',
    '192暂无': 'None3',
    # 8. 获利能力分析
    '193成本费用利润率(%)': 'rateOfReturnOnCost',
    '194营业利润率': 'rateOfReturnOnOperatingProfit',
    '195营业税金率': 'rateOfReturnOnBusinessTax',
    '196营业成本率': 'rateOfReturnOnOperatingCost',
    '197净资产收益率': 'rateOfReturnOnCommonStockholdersEquity',
    '198投资收益率': 'rateOfReturnOnInvestmentIncome',
    '199销售净利率(%)': 'rateOfReturnOnNetSalesProfit',
    '200总资产报酬率': 'rateOfReturnOnTotalAssets',
    '201净利润率': 'netProfitMargin',
    '202销售毛利率(%)': 'rateOfReturnOnGrossProfitFromSales',
    '203三费比重': 'threeFeeProportion',
    '204管理费用率': 'ratioOfChargingExpense',
    '205财务费用率': 'ratioOfFinancialExpense',
    '206扣除非经常性损益后的净利润': 'netProfitAfterExtraordinaryGainsAndLosses',
    '207息税前利润(EBIT)': 'EBIT',
    '208息税折旧摊销前利润(EBITDA)': 'EBITDA',
    '209EBITDA/营业总收入(%)': 'EBITDA/GrossRevenueRate',
    # 9. 资本结构分析
    '210资产负债率(%)': 'assetsLiabilitiesRatio',
    '211流动资产比率': 'currentAssetsRatio',  # 期末的流动资产除以所有者权益
    '212货币资金比率': 'monetaryFundRatio',
    '213存货比率': 'inventoryRatio',
    '214固定资产比率': 'fixedAssetsRatio',
    '215负债结构比': 'liabilitiesStructureRatio',
    '216归属于母公司股东权益/全部投入资本(%)': 'shareholdersOwnershipOfAParentCompany/TotalCapital',
    '217股东的权益/带息债务(%)': 'shareholdersInterest/InterestRateDebtRatio',
    '218有形资产/净债务(%)': 'tangibleAssets/NetDebtRatio',
    # 10. 现金流量分析
    '219每股经营性现金流(元)': 'operatingCashFlowPerShare',
    '220营业收入现金含量(%)': 'cashOfOperatingIncome',
    '221经营活动产生的现金流量净额/经营活动净收益(%)': 'netOperatingCashFlow/netOperationProfit',
    '222销售商品提供劳务收到的现金/营业收入(%)': 'cashFromGoodsSales/OperatingRevenue',
    '223经营活动产生的现金流量净额/营业收入': 'netOperatingCashFlow/OperatingRevenue',
    '224资本支出/折旧和摊销': 'capitalExpenditure/DepreciationAndAmortization',
    '225每股现金流量净额(元)': 'netCashFlowPerShare',
    '226经营净现金比率（短期债务）': 'operatingCashFlow/ShortTermDebtRatio',
    '227经营净现金比率（全部债务）': 'operatingCashFlow/LongTermDebtRatio',
    '228经营活动现金净流量与净利润比率': 'cashFlowRateAndNetProfitRatioOfOperatingActivities',
    '229全部资产现金回收率': 'cashRecoveryForAllAssets',
    # 11. 单季度财务指标
    '230营业收入': 'operatingRevenueSingle',
    '231营业利润': 'operatingProfitSingle',
    '232归属于母公司所有者的净利润': 'netProfitBelongingToTheOwnerOfTheParentCompanySingle',
    '233扣除非经常性损益后的净利润': 'netProfitAfterExtraordinaryGainsAndLossesSingle',
    '234经营活动产生的现金流量净额': 'netCashFlowsFromOperatingActivitiesSingle',
    '235投资活动产生的现金流量净额': 'netCashFlowsFromInvestingActivitiesSingle',
    '236筹资活动产生的现金流量净额': 'netCashFlowsFromFinancingActivitiesSingle',
    '237现金及现金等价物净增加额': 'netIncreaseInCashAndCashEquivalentsSingle',
    # 12.股本股东
    '238总股本': 'totalCapital',
    '239已上市流通A股': 'listedAShares',
    '240已上市流通B股': 'listedBShares',
    '241已上市流通H股': 'listedHShares',
    '242股东人数(户)': 'numberOfShareholders',
    '243第一大股东的持股数量': 'theNumberOfFirstMajorityShareholder',
    '244十大流通股东持股数量合计(股)': 'totalNumberOfTopTenCirculationShareholders',
    '245十大股东持股数量合计(股)': 'totalNumberOfTopTenMajorShareholders',
    # 13.机构持股
    '246机构总量（家）': 'institutionNumber',
    '247机构持股总量(股)': 'institutionShareholding',
    '248QFII机构数': 'QFIIInstitutionNumber',
    '249QFII持股量': 'QFIIShareholding',
    '250券商机构数': 'brokerNumber',
    '251券商持股量': 'brokerShareholding',
    '252保险机构数': 'securityNumber',
    '253保险持股量': 'securityShareholding',
    '254基金机构数': 'fundsNumber',
    '255基金持股量': 'fundsShareholding',
    '256社保机构数': 'socialSecurityNumber',
    '257社保持股量': 'socialSecurityShareholding',
    '258私募机构数': 'privateEquityNumber',
    '259私募持股量': 'privateEquityShareholding',
    '260财务公司机构数': 'financialCompanyNumber',
    '261财务公司持股量': 'financialCompanyShareholding',
    '262年金机构数': 'pensionInsuranceAgencyNumber',
    '263年金持股量': 'pensionInsuranceAgencyShareholfing',
    # 14.新增指标
    # [注：季度报告中，若股东同时持有非流通A股性质的股份(如同时持有流通A股和流通B股），取的是包含同时持有非流通A股性质的流通股数]
    '264十大流通股东中持有A股合计(股)': 'totalNumberOfTopTenCirculationShareholders',
    '265第一大流通股东持股量(股)': 'firstLargeCirculationShareholdersNumber',
    # [注：1.自由流通股=已流通A股-十大流通股东5%以上的A股；2.季度报告中，若股东同时持有非流通A股性质的股份(如同时持有流通A股和流通H股），5%以上的持股取的是不包含同时持有非流通A股性质的流通股数，结果可能偏大； 3.指标按报告期展示，新股在上市日的下个报告期才有数据]
    '266自由流通股(股)': 'freeCirculationStock',
    '267受限流通A股(股)': 'limitedCirculationAShares',
    '268一般风险准备(金融类)': 'generalRiskPreparation',
    '269其他综合收益(利润表)': 'otherComprehensiveIncome',
    '270综合收益总额(利润表)': 'totalComprehensiveIncome',
    '271归属于母公司股东权益(资产负债表)': 'shareholdersOwnershipOfAParentCompany ',
    '272银行机构数(家)(机构持股)': 'bankInstutionNumber',
    '273银行持股量(股)(机构持股)': 'bankInstutionShareholding',
    '274一般法人机构数(家)(机构持股)': 'corporationNumber',
    '275一般法人持股量(股)(机构持股)': 'corporationShareholding',
    '276近一年净利润(元)': 'netProfitLastYear',
    '277信托机构数(家)(机构持股)': 'trustInstitutionNumber',
    '278信托持股量(股)(机构持股)': 'trustInstitutionShareholding',
    '279特殊法人机构数(家)(机构持股)': 'specialCorporationNumber',
    '280特殊法人持股量(股)(机构持股)': 'specialCorporationShareholding',
    '281加权净资产收益率(每股指标)': 'weightedROE',
    '282扣非每股收益(单季度财务指标)': 'nonEPSSingle',
    '283最近一年营业收入(万元)': 'lastYearOperatingIncome',
    '284国家队持股数量(万股)': 'nationalTeamShareholding',
    # [注：本指标统计包含汇金公司、证金公司、外汇管理局旗下投资平台、国家队基金、国开、养老金以及中科汇通等国家队机构持股数量]
    '285业绩预告-本期净利润同比增幅下限%': 'PF_theLowerLimitoftheYearonyearGrowthofNetProfitForThePeriod',
    # [注：指标285至294展示未来一个报告期的数据。例，3月31日至6月29日这段时间内展示的是中报的数据；如果最新的财务报告后面有多个报告期的业绩预告/快报，只能展示最新的财务报告后面的一个报告期的业绩预告/快报]
    '286业绩预告-本期净利润同比增幅上限%': 'PF_theHigherLimitoftheYearonyearGrowthofNetProfitForThePeriod',
    '287业绩快报-归母净利润': 'PE_returningtotheMothersNetProfit',
    '288业绩快报-扣非净利润': 'PE_Non-netProfit',
    '289业绩快报-总资产': 'PE_TotalAssets',
    '290业绩快报-净资产': 'PE_NetAssets',
    '291业绩快报-每股收益': 'PE_EPS',
    '292业绩快报-摊薄净资产收益率': 'PE_DilutedROA',
    '293业绩快报-加权净资产收益率': 'PE_WeightedROE',
    '294业绩快报-每股净资产': 'PE_NetAssetsperShare',
    '295应付票据及应付账款(资产负债表)': 'BS_NotesPayableandAccountsPayable',
    '296应收票据及应收账款(资产负债表)': 'BS_NotesReceivableandAccountsReceivable',
    '297递延收益(资产负债表)': 'BS_DeferredIncome',
    '298其他综合收益(资产负债表)': 'BS_OtherComprehensiveIncome',
    '299其他权益工具(资产负债表)': 'BS_OtherEquityInstruments',
    '300其他收益(利润表)': 'IS_OtherIncome',
    '301资产处置收益(利润表)': 'IS_AssetDisposalIncome',
    '302持续经营净利润(利润表)': 'IS_NetProfitforContinuingOperations',
    '303终止经营净利润(利润表)': 'IS_NetProfitforTerminationOperations',
    '304研发费用(利润表)': 'IS_R&DExpense',
    '305其中:利息费用(利润表-财务费用)': 'IS_InterestExpense',
    '306其中:利息收入(利润表-财务费用)': 'IS_InterestIncome',
    '307近一年经营活动现金流净额': 'netCashFlowfromOperatingActivitiesinthepastyear',
    '308未知308': 'unknown308',
    '309未知309': 'unknown309',
    '310未知310': 'unknown310',
    '311未知311': 'unknown311',
    '312未知312': 'unknown312',
    '313未知313': 'unknown313',
    '314未知314': 'unknown314',
    '315未知315': 'unknown315',
    '316未知316': 'unknown316',
    '317未知317': 'unknown317',
    '318未知318': 'unknown318',

    '319未知319': 'unknown319',
    '320未知320': 'unknown320',
    '321未知321': 'unknown321',
    '322未知322': 'unknown322',
    '323未知323': 'unknown323',
    '324未知324': 'unknown324',
    '325未知325': 'unknown325',
    '326未知326': 'unknown326',
    '327未知327': 'unknown327',
    '328未知328': 'unknown328',
    '329未知329': 'unknown329',
    '330未知330': 'unknown330',
    '331未知331': 'unknown331',
    '332未知332': 'unknown332',
    '333未知333': 'unknown333',
    '334未知334': 'unknown334',
    '335未知335': 'unknown335',
    '336未知336': 'unknown336',
    '337未知337': 'unknown337',
    '338未知338': 'unknown338',
    '339未知339': 'unknown339',
    '340未知340': 'unknown340',
    '341未知341': 'unknown341',
    '342未知342': 'unknown342',
    '343未知343': 'unknown343',
    '344未知344': 'unknown344',
    '345未知345': 'unknown345',
    '346未知346': 'unknown346',
    '347未知347': 'unknown347',
    '348未知348': 'unknown348',
    '349未知349': 'unknown349',
    '350未知350': 'unknown350',
    '351未知351': 'unknown351',
    '352未知352': 'unknown352',
    '353未知353': 'unknown353',
    '354未知354': 'unknown354',
    '355未知355': 'unknown355',
    '356未知356': 'unknown356',
    '357未知357': 'unknown357',
    '358未知358': 'unknown358',
    '359未知359': 'unknown359',
    '360未知360': 'unknown360',
    '361未知361': 'unknown361',
    '362未知362': 'unknown362',
    '363未知363': 'unknown363',
    '364未知364': 'unknown364',
    '365未知365': 'unknown365',
    '366未知366': 'unknown366',
    '367未知367': 'unknown367',
    '368未知368': 'unknown368',
    '369未知369': 'unknown369',
    '370未知370': 'unknown370',
    '371未知371': 'unknown371',
    '372未知372': 'unknown372',
    '373未知373': 'unknown373',
    '374未知374': 'unknown374',
    '375未知375': 'unknown375',
    '376未知376': 'unknown376',
    '377未知377': 'unknown377',
    '378未知378': 'unknown378',
    '379未知379': 'unknown379',
    '380未知380': 'unknown380',
    '381未知381': 'unknown381',
    '382未知382': 'unknown382',
    '383未知383': 'unknown383',
    '384未知384': 'unknown384',
    '385未知385': 'unknown385',
    '386未知386': 'unknown386',
    '387未知387': 'unknown387',
    '388未知388': 'unknown388',
    '389未知389': 'unknown389',
    '390未知390': 'unknown390',
    '391未知391': 'unknown391',
    '392未知392': 'unknown392',
    '393未知393': 'unknown393',
    '394未知394': 'unknown394',
    '395未知395': 'unknown395',
    '396未知396': 'unknown396',
    '397未知397': 'unknown397',
    '398未知398': 'unknown398',
    '399未知399': 'unknown399',
    '400未知400': 'unknown400',
    '401未知401': 'unknown401',
    '402未知402': 'unknown402',
    '403未知403': 'unknown403',
    '404未知404': 'unknown404',
    '405未知405': 'unknown405',
    '406未知406': 'unknown406',
    '407未知407': 'unknown407',
    '408未知408': 'unknown408',
    '409未知409': 'unknown409',
    '410未知410': 'unknown410',
    '411未知411': 'unknown411',
    '412未知412': 'unknown412',
    '413未知413': 'unknown413',
    '414未知414': 'unknown414',
    '415未知415': 'unknown415',
    '416未知416': 'unknown416',
    '417未知417': 'unknown417',
    '418未知418': 'unknown418',
    '419未知419': 'unknown419',
    '420未知420': 'unknown420',
    '421未知421': 'unknown421',
    '422未知422': 'unknown422',
    '423未知423': 'unknown423',
    '424未知424': 'unknown424',
    '425未知425': 'unknown425',
    '426未知426': 'unknown426',
    '427未知427': 'unknown427',
    '428未知428': 'unknown428',
    '429未知429': 'unknown429',
    '430未知430': 'unknown430',
    '431未知431': 'unknown431',
    '432未知432': 'unknown432',
    '433未知433': 'unknown433',
    '434未知434': 'unknown434',
    '435未知435': 'unknown435',
    '436未知436': 'unknown436',
    '437未知437': 'unknown437',
    '438未知438': 'unknown438',
    '439未知439': 'unknown439',
    '440未知440': 'unknown440',
    '441未知441': 'unknown441',
    '442未知442': 'unknown442',
    '443未知443': 'unknown443',
    '444未知444': 'unknown444',
    '445未知445': 'unknown445',
    '446未知446': 'unknown446',
    '447未知447': 'unknown447',
    '448未知448': 'unknown448',
    '449未知449': 'unknown449',
    '450未知450': 'unknown450',
    '451未知451': 'unknown451',
    '452未知452': 'unknown452',
    '453未知453': 'unknown453',
    '454未知454': 'unknown454',
    '455未知455': 'unknown455',
    '456未知456': 'unknown456',
    '457未知457': 'unknown457',
    '458未知458': 'unknown458',
    '459未知459': 'unknown459',
    '460未知460': 'unknown460',
    '461未知461': 'unknown461',
    '462未知462': 'unknown462',
    '463未知463': 'unknown463',
    '464未知464': 'unknown464',
    '465未知465': 'unknown465',
    '466未知466': 'unknown466',
    '467未知467': 'unknown467',
    '468未知468': 'unknown468',
    '469未知469': 'unknown469',
    '470未知470': 'unknown470',
    '471未知471': 'unknown471',
    '472未知472': 'unknown472',
    '473未知473': 'unknown473',
    '474未知474': 'unknown474',
    '475未知475': 'unknown475',
    '476未知476': 'unknown476',
    '477未知477': 'unknown477',
    '478未知478': 'unknown478',
    '479未知479': 'unknown479',
    '480未知480': 'unknown480',
    '481未知481': 'unknown481',
    '482未知482': 'unknown482',
    '483未知483': 'unknown483',
    '484未知484': 'unknown484',
    '485未知485': 'unknown485',
    '486未知486': 'unknown486',
    '487未知487': 'unknown487',
    '488未知488': 'unknown488',
    '489未知489': 'unknown489',
    '490未知490': 'unknown490',
    '491未知491': 'unknown491',
    '492未知492': 'unknown492',
    '493未知493': 'unknown493',
    '494未知494': 'unknown494',
    '495未知495': 'unknown495',
    '496未知496': 'unknown496',
    '497未知497': 'unknown497',
    '498未知498': 'unknown498',
    '499未知499': 'unknown499',
    '500未知500': 'unknown500',
    '501未知501': 'unknown501',
    '502未知502': 'unknown502',
    '503未知503': 'unknown503',
    '504未知504': 'unknown504',
    '505未知505': 'unknown505',
    '506未知506': 'unknown506',
    '507未知507': 'unknown507',
    '508未知508': 'unknown508',
    '509未知509': 'unknown509',
    '510未知510': 'unknown510',
    '511未知511': 'unknown511',
    '512未知512': 'unknown512',
    '513未知513': 'unknown513',
    '514未知514': 'unknown514',
    '515未知515': 'unknown515',
    '516未知516': 'unknown516',
    '517未知517': 'unknown517',
    '518未知518': 'unknown518',
    '519未知519': 'unknown519',
    '520未知520': 'unknown520',
    '521未知521': 'unknown521',
    '522未知522': 'unknown522',
    '523未知523': 'unknown523',
    '524未知524': 'unknown524',
    '525未知525': 'unknown525',
    '526未知526': 'unknown526',
    '527未知527': 'unknown527',
    '528未知528': 'unknown528',
    '529未知529': 'unknown529',
    '530未知530': 'unknown530',
    '531未知531': 'unknown531',
    '532未知532': 'unknown532',
    '533未知533': 'unknown533',
    '534未知534': 'unknown534',
    '535未知535': 'unknown535',
    '536未知536': 'unknown536',
    '537未知537': 'unknown537',
    '538未知538': 'unknown538',
    '539未知539': 'unknown539',
    '540未知540': 'unknown540',
    '541未知541': 'unknown541',
    '542未知542': 'unknown542',
    '543未知543': 'unknown543',
    '544未知544': 'unknown544',
    '545未知545': 'unknown545',
    '546未知546': 'unknown546',
    '547未知547': 'unknown547',
    '548未知548': 'unknown548',
    '549未知549': 'unknown549',
    '550未知550': 'unknown550',
    '551未知551': 'unknown551',
    '552未知552': 'unknown552',
    '553未知553': 'unknown553',
    '554未知554': 'unknown554',
    '555未知555': 'unknown555',
    '556未知556': 'unknown556',
    '557未知557': 'unknown557',
    '558未知558': 'unknown558',
    '559未知559': 'unknown559',
    '560未知560': 'unknown560',
    '561未知561': 'unknown561',
    '562未知562': 'unknown562',
    '563未知563': 'unknown563',
    '564未知564': 'unknown564',
    '565未知565': 'unknown565',
    '566未知566': 'unknown566',
    '567未知567': 'unknown567',
    '568未知568': 'unknown568',
    '569未知569': 'unknown569',
    '570未知570': 'unknown570',
    '571未知571': 'unknown571',
    '572未知572': 'unknown572',
    '573未知573': 'unknown573',
    '574未知574': 'unknown574',
    '575未知575': 'unknown575',
    '576未知576': 'unknown576',
    '577未知577': 'unknown577',
    '578未知578': 'unknown578',
    '579未知579': 'unknown579',
    '580未知580': 'unknown580',

}
