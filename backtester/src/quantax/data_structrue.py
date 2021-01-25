# coding:utf-8

"""
定义一些可以扩展的数据结构

方便序列化/相互转换

"""

from copy import copy
from functools import lru_cache

import pandas as pd

from quantax.base_data_structure import _quotation_base
from quantax.data_fq import QA_data_stock_to_fq
from quantax.data_resample import (
    tick_resample,
    day_resample,
    futureday_resample,
    min_resample,
    futuremin_resample
)
from quantax.settings import DATABASE
from quantax.utilities import (
    QA_util_to_json_from_pandas,
    QA_util_date_valid,
    QA_util_code_tolist,
    financial_dict
)


def _QA_fetch_stock_adj(
        code,
        start,
        end,
        format='pd',
        collections=DATABASE.stock_adj
):
    """获取股票复权系数 ADJ

    """

    start = str(start)[0:10]
    end = str(end)[0:10]
    # code= [code] if isinstance(code,str) else code

    # code checking
    code = QA_util_code_tolist(code)

    if QA_util_date_valid(end):
        cursor = collections.find(
            {
                'code': {
                    '$in': code
                },
                "date": {
                    "$lte": end,
                    "$gte": start
                }
            },
            {"_id": 0},
            batch_size=10000
        )
        # res=[QA_util_dict_remove_key(data, '_id') for data in cursor]

        res = pd.DataFrame([item for item in cursor])
        res.date = pd.to_datetime(res.date)
        return res.set_index('date', drop=False)


class StockDay(_quotation_base):
    '''

        股票日线数据
    '''

    def __init__(self, init_data_by_df, dtype='stock_day', if_fq='bfq'):
        '''
        # 🛠 todo dtype=stock_day 和 QA_DataStruct_Stock_day 类的名字是对应的 不变的不需要指定 ，容易出错，建议改成常量 ❌
        :param init_data_by_df:  DataFrame 类型的数据，包含了数据，用来初始化这个类
        :param dtype:  stock_day 🛠 todo 改成常量
        :param if_fq:  是否复权
        '''
        super().__init__(init_data_by_df, dtype, if_fq)

        if isinstance(init_data_by_df, pd.DataFrame) == False:
            print("QAError init_data_by_df is not kind of DataFrame type !")

    # 抽象类继承

    def choose_db(self):
        self.mongo_coll = DATABASE.stock_day

    def __repr__(self):
        return '< QA_DataStruct_Stock_day with {} securities >'.format(
            len(self.code)
        )

    __str__ = __repr__

    # 前复权
    def to_qfq(self):
        if self.if_fq == 'bfq':
            if len(self.code) < 1:
                self.if_fq = 'qfq'
                return self
            # elif len(self.code) < 20:
            #     return self.new(pd.concat(list(map(
            #         lambda x: QA_data_stock_to_fq(self.data[self.data['code'] == x]), self.code))), self.type, 'qfq')
            else:
                try:
                    date = self.date
                    adj = _QA_fetch_stock_adj(
                        self.code.to_list(),
                        str(date[0])[0:10],
                        str(date[-1])[0:10]
                    ).set_index(['date',
                                 'code'])
                    data = self.data.join(adj)
                    for col in ['open', 'high', 'low', 'close']:
                        data[col] = data[col] * data['adj']
                    # data['volume'] = data['volume'] / \
                    #     data['adj'] if 'volume' in data.columns else data['vol']/data['adj']

                    data['volume'] = data['volume'] if 'volume' in data.columns else data['vol']
                    try:
                        data['high_limit'] = data['high_limit'] * data['adj']
                        data['low_limit'] = data['high_limit'] * data['adj']
                    except:
                        pass
                    return self.new(data, self.type, 'qfq')
                except Exception as e:
                    print(e)
                    print('use old model qfq')
                    return self.new(
                        self.groupby(level=1).apply(QA_data_stock_to_fq,
                                                    'qfq'),
                        self.type,
                        'qfq'
                    )
        else:
            print(
                'none support type for qfq Current type is: %s' % self.if_fq
            )
            return self

    # 后复权
    def to_hfq(self):
        if self.if_fq == 'bfq':
            if len(self.code) < 1:
                self.if_fq = 'hfq'
                return self
            else:
                return self.new(
                    self.groupby(level=1).apply(QA_data_stock_to_fq,
                                                'hfq'),
                    self.type,
                    'hfq'
                )
                # return self.new(pd.concat(list(map(lambda x: QA_data_stock_to_fq(
                #     self.data[self.data['code'] == x], 'hfq'), self.code))), self.type, 'hfq')
        else:
            print(
                'none support type for qfq Current type is: %s' % self.if_fq
            )
            return self

    @property
    @lru_cache()
    def high_limit(self):
        '涨停价'
        return self.groupby(
            level=1
        ).close.apply(lambda x: round((x.shift(1) + 0.0002) * 1.1,
                                      2)).sort_index()

    @property
    @lru_cache()
    def low_limit(self):
        '跌停价'
        return self.groupby(
            level=1
        ).close.apply(lambda x: round((x.shift(1) + 0.0002) * 0.9,
                                      2)).sort_index()

    @property
    @lru_cache()
    def next_day_low_limit(self):
        "明日跌停价"
        return self.groupby(
            level=1
        ).close.apply(lambda x: round((x + 0.0002) * 0.9,
                                      2)).sort_index()

    @property
    @lru_cache()
    def next_day_high_limit(self):
        "明日涨停价"
        return self.groupby(
            level=1
        ).close.apply(lambda x: round((x + 0.0002) * 1.1,
                                      2)).sort_index()

    @property
    def preclose(self):
        try:
            return self.data.preclose
        except:
            return None

    pre_close = preclose

    @property
    def price_chg(self):
        try:
            return (self.close - self.preclose) / self.preclose
        except:
            return None

    @property
    @lru_cache()
    def week(self):
        return self.resample('w')

    @property
    @lru_cache()
    def month(self):
        return self.resample('M')

    @property
    @lru_cache()
    def quarter(self):
        return self.resample('Q')

    # @property
    # @lru_cache()
    # def semiannual(self):
    #     return self.resample('SA')

    @property
    @lru_cache()
    def year(self):
        return self.resample('Y')

    def resample(self, level):
        try:
            return self.add_func(day_resample, level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None


class StockMin(_quotation_base):

    def __init__(self, DataFrame, dtype='stock_min', if_fq='bfq'):
        super().__init__(DataFrame, dtype, if_fq)

        try:
            if 'preclose' in DataFrame.columns:
                self.data = DataFrame.loc[:,
                            [
                                'open',
                                'high',
                                'low',
                                'close',
                                'volume',
                                'amount',
                                'preclose',
                                'type'
                            ]]
            else:
                self.data = DataFrame.loc[:,
                            [
                                'open',
                                'high',
                                'low',
                                'close',
                                'volume',
                                'amount',
                                'type'
                            ]]
        except Exception as e:
            raise e

        self.type = dtype
        self.if_fq = if_fq

        self.data = self.data.sort_index()

    # 抽象类继承
    def choose_db(self):
        self.mongo_coll = DATABASE.stock_min

    def __repr__(self):
        return '< QA_DataStruct_Stock_Min with {} securities >'.format(
            len(self.code)
        )

    __str__ = __repr__

    def to_qfq(self):
        if self.if_fq == 'bfq':
            if len(self.code) < 1:
                self.if_fq = 'qfq'
                return self
            # elif len(self.code) < 20:
            #     data = QA_DataStruct_Stock_min(pd.concat(list(map(lambda x: QA_data_stock_to_fq(
            #         self.data[self.data['code'] == x]), self.code))).set_index(['datetime', 'code'], drop=False))
            #     data.if_fq = 'qfq'
            #     return data
            else:
                try:
                    date = self.date
                    adj = _QA_fetch_stock_adj(
                        self.code.to_list(),
                        str(date[0])[0:10],
                        str(date[-1])[0:10]
                    ).set_index(['date',
                                 'code'])
                    u = self.data.reset_index()
                    u = u.assign(date=u.datetime.apply(lambda x: x.date()))
                    u = u.set_index(['date', 'code'], drop=False)

                    data = u.join(adj).set_index(['datetime', 'code'])

                    for col in ['open', 'high', 'low', 'close']:
                        data[col] = data[col] * data['adj']
                    # data['volume'] = data['volume'] / \
                    #     data['adj']
                    # data['volume'] = data['volume']  if 'volume' in data.columns else data['vol']
                    try:
                        data['high_limit'] = data['high_limit'] * data['adj']
                        data['low_limit'] = data['high_limit'] * data['adj']
                    except:
                        pass
                    return self.new(data, self.type, 'qfq')
                except Exception as e:
                    print(e)
                    print('use old model qfq')
                    return self.new(
                        self.groupby(level=1).apply(QA_data_stock_to_fq,
                                                    'qfq'),
                        self.type,
                        'qfq'
                    )

        else:
            print(
                'none support type for qfq Current type is:%s' % self.if_fq
            )
            return self

    def to_hfq(self):
        if self.if_fq == 'bfq':
            if len(self.code) < 1:
                self.if_fq = 'hfq'
                return self
            else:
                return self.new(
                    self.groupby(level=1).apply(QA_data_stock_to_fq,
                                                'hfq'),
                    self.type,
                    'hfq'
                )
                # data = QA_DataStruct_Stock_min(pd.concat(list(map(lambda x: QA_data_stock_to_fq(
                #     self.data[self.data['code'] == x], 'hfq'), self.code))).set_index(['datetime', 'code'], drop=False))
                # data.if_fq = 'hfq'
                # return data
        else:
            print(
                'none support type for qfq Current type is:%s' % self.if_fq
            )
            return self

    # @property
    # def high_limit(self):
    #     '涨停价'
    #     return self.data.high_limit

    # @property
    # def low_limit(self):
    #     '跌停价'
    #     return self.data.low_limit

    def resample(self, level):
        try:
            return self.add_funcx(min_resample, level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None

    @property
    @lru_cache()
    def min5(self):
        return self.resample('5min')

    @property
    @lru_cache()
    def min15(self):
        return self.resample('15min')

    @property
    @lru_cache()
    def min30(self):
        return self.resample('30min')

    @property
    @lru_cache()
    def min60(self):
        return self.resample('60min')


class IndexMin(_quotation_base):
    '自定义的分钟线数据结构'

    def __init__(self, DataFrame, dtype='index_min', if_fq=''):
        super().__init__(DataFrame, dtype, if_fq)

        try:
            if 'preclose' in DataFrame.columns:
                self.data = DataFrame.loc[:,
                            [
                                'open',
                                'high',
                                'low',
                                'close',
                                'volume',
                                'amount',
                                'preclose',
                                'type'
                            ]]
            else:
                self.data = DataFrame.loc[:,
                            [
                                'open',
                                'high',
                                'low',
                                'close',
                                'volume',
                                'amount',
                                'type'
                            ]]
        except Exception as e:
            raise e

        self.type = dtype
        self.if_fq = if_fq

        self.data = self.data.sort_index()

    # 抽象类继承
    def choose_db(self):
        self.mongo_coll = DATABASE.index_min

    def __repr__(self):
        return '< QA_DataStruct_Index_Min with {} instruments>'.format(
            len(self.code)
        )

    @property
    @lru_cache()
    def min5(self):
        return self.resample('5min')

    @property
    @lru_cache()
    def min15(self):
        return self.resample('15min')

    @property
    @lru_cache()
    def min30(self):
        return self.resample('30min')

    @property
    @lru_cache()
    def min60(self):
        return self.resample('60min')

    def resample(self, level):
        try:
            return self.add_funcx(min_resample, level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None

    __str__ = __repr__


class FutureDay(_quotation_base):

    def __init__(self, DataFrame, dtype='future_day', if_fq=''):
        super().__init__(DataFrame, dtype, if_fq)
        self.type = 'future_day'
        self.data = self.data.loc[:,
                    [
                        'open',
                        'high',
                        'low',
                        'close',
                        'volume',
                        'position',
                        'price'
                    ]]
        self.if_fq = if_fq

    # 抽象类继承
    def choose_db(self):
        self.mongo_coll = DATABASE.future_day

    def __repr__(self):
        return '< QA_DataStruct_Future_day with {} securities >'.format(
            len(self.code)
        )

    __str__ = __repr__

    @property
    @lru_cache()
    def week(self):
        return self.resample('w')

    @property
    @lru_cache()
    def month(self):
        return self.resample('M')

    @property
    @lru_cache()
    def quarter(self):
        return self.resample('Q')

    @property
    @lru_cache()
    def tradedate(self):
        """返回交易所日历下的日期

        Returns:
            [type] -- [description]
        """

        try:
            return self.date
        except:
            return None

    @property
    @lru_cache()
    def tradetime(self):
        """返回交易所日历下的日期

        Returns:
            [type] -- [description]
        """

        try:
            return self.date
        except:
            return None

    # @property
    # @lru_cache()
    # def semiannual(self):
    #     return self.resample('SA')

    @property
    @lru_cache()
    def year(self):
        return self.resample('Y')

    def resample(self, level):
        try:
            return self.add_func(futureday_resample, level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None


class FutureMin(_quotation_base):
    """
    struct for future
    """

    def __init__(self, DataFrame, dtype='future_min', if_fq=''):
        # 🛠todo  期货分钟数据线的维护， 暂时用日线代替分钟线
        super().__init__(DataFrame, dtype, if_fq)
        self.type = dtype
        self.data = self.data.loc[:,
                    [
                        'open',
                        'high',
                        'low',
                        'close',
                        'volume',
                        'position',
                        'price',
                        'tradetime',
                        'type'
                    ]]
        self.if_fq = if_fq

    # 抽象类继承
    def choose_db(self):
        self.mongo_coll = DATABASE.future_min

    @property
    @lru_cache()
    def tradedate(self):
        """返回交易所日历下的日期

        Returns:
            [type] -- [description]
        """

        try:
            return self.data.tradetime.apply(lambda x: x[0:10])
        except:
            return None

    @property
    @lru_cache()
    def tradetime(self):
        """返回交易所日历下的日期

        Returns:
            [type] -- [description]
        """

        try:
            return self.data.tradetime
        except:
            return None

    @property
    @lru_cache()
    def min5(self):
        return self.resample('5min')

    @property
    @lru_cache()
    def min15(self):
        return self.resample('15min')

    @property
    @lru_cache()
    def min30(self):
        return self.resample('30min')

    @property
    @lru_cache()
    def min60(self):
        return self.resample('60min')

    def __repr__(self):
        return '< QA_DataStruct_Future_min with {} securities >'.format(
            len(self.code)
        )

    __str__ = __repr__

    def resample(self, level):
        try:
            return self.add_funcx(futuremin_resample,
                                  level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None


class IndexDay(_quotation_base):
    '自定义的日线数据结构'

    def __init__(self, DataFrame, dtype='index_day', if_fq=''):
        super().__init__(DataFrame, dtype, if_fq)
        # self.data = DataFrame
        self.type = dtype
        self.if_fq = if_fq
        # self.mongo_coll = eval(
        #    'DATABASE.{}'.format(self.type))

    """
    def __add__(self,DataStruct):
        'add func with merge list and reindex'
        assert isinstance(DataStruct,QA_DataStruct_Index_day)
        if self.if_fq==DataStruct.if_fq:
            self.sync_status(pd.concat())
    """

    # 抽象类继承
    def choose_db(self):
        self.mongo_coll = DATABASE.index_day

    def __repr__(self):
        return '< QA_DataStruct_Index_day with {} securities >'.format(
            len(self.code)
        )

    __str__ = __repr__

    @property
    @lru_cache()
    def week(self):
        return self.resample('w')

    @property
    @lru_cache()
    def month(self):
        return self.resample('M')

    @property
    @lru_cache()
    def quarter(self):
        return self.resample('Q')

    # @property
    # @lru_cache()
    # def semiannual(self):
    #     return self.resample('SA')

    @property
    @lru_cache()
    def year(self):
        return self.resample('Y')

    def resample(self, level):
        try:
            return self.add_func(day_resample, level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None


class StockTransaction():

    def __init__(self, DataFrame):
        """Stock Transaction

        Arguments:
            DataFrame {pd.Dataframe} -- [input is one/multi day transaction]
        """

        self.type = 'stock_transaction'

        self.data = DataFrame
        if 'amount' not in DataFrame.columns:
            if 'vol' in DataFrame.columns:
                self.data['amount'] = self.data.vol * self.data.price * 100
            elif 'volume' in DataFrame.columns:
                self.data['amount'] = self.data.volume * self.data.price * 100
        if '_id' in DataFrame.columns:
            self.data = self.data.drop(["_id"], axis=1)
        self.mongo_coll = DATABASE.stock_transaction

    @property
    @lru_cache()
    def buyorsell(self):
        """return the buy or sell towards 0--buy 1--sell 2--none

        Decorators:
            lru_cache

        Returns:
            [pd.Series] -- [description]
        """

        return self.data.buyorsell

    @property
    @lru_cache()
    def price(self):
        """return the deal price of tick transaction

        Decorators:
            lru_cache

        Returns:
            [type] -- [description]
        """

        return self.data.price

    @property
    @lru_cache()
    def vol(self):
        """return the deal volume of tick

        Decorators:
            lru_cache

        Returns:
            pd.Series -- volume of transaction
        """

        try:
            return self.data.volume
        except:
            return self.data.vol

    volume = vol

    @property
    @lru_cache()
    def date(self):
        """return the date of transaction

        Decorators:
            lru_cache

        Returns:
            pd.Series -- date of transaction
        """

        return self.data.date

    @property
    @lru_cache()
    def time(self):
        """return the exact time of transaction(to minute level)

        Decorators:
            lru_cache

        Returns:
            pd.Series -- till minute level
        """

        return self.data.time

    @property
    @lru_cache()
    def datetime(self):
        """return the datetime of transaction

        Decorators:
            lru_cache

        Returns:
            pd.Series -- [description]
        """

        return self.data.datetime

    @property
    @lru_cache()
    def order(self):
        """return the order num of transaction/ for everyday change

        Decorators:
            lru_cache

        Returns:
            pd.series -- [description]
        """

        return self.data.order

    @property
    @lru_cache()
    def index(self):
        """return the transaction index

        Decorators:
            lru_cache

        Returns:
            [type] -- [description]
        """

        return self.data.index

    @property
    @lru_cache()
    def amount(self):
        """return current tick trading amount

        Decorators:
            lru_cache

        Returns:
            [type] -- [description]
        """

        return self.data.amount

    """
    最新:IF(ISNULL(NEW),PRE,NEW);
    IF (ISNULL(RANGE_AVG_PRICE) OR RANGE_AVG_PRICE <= 0)
    {
        IF (MARKETTYPE == 232 OR MARKETTYPE == 56 OR MARKETTYPE==64 OR MARKETTYPE==128 OR MARKETTYPE==168 OR MARKETTYPE==184 OR MARKETTYPE == 200 OR MARKETTYPE == 80 OR (VOL > 1 AND VOL<100))
        {
            b=SUBSAMEDAY(&VOL) ;
            m=SUM(b*最新,0);
            均价:IF(m>0,m/VOL,PRE);
        }
        ELSE IF(CODETYPE!=0 AND MONEY>0)
        {
            IF(ISNULL(MONEY) OR ISNULL(VOL) OR VOL==0 OR MONEY==0)
                均价:PRE;
            ELSE IF(VOL==VOL[1] OR MONEY==MONEY[1])
                均价:均价[1];
            ELSE
                均价:MONEY/VOL;
        }
        ELSE IF (MARKETTYPE == 176)
        {
            b=SUBSAMEDAY(&MONEY);
            m=SUM(b*最新,0);
            IF(m>0)
                均价:m/MONEY;
        }
    }
    ELSE
    {
        均价:RANGE_AVG_PRICE;
    }
    DRAWGBK(MARKETTYPE==32 AND FORMATTIME(1)<10 AND TRADETIME>242),RGB(0,0,128);
    RETURN;


    hx_star;
    hx_star_p;
    """

    def __repr__(self):
        return '< QA_DataStruct_Stock_Transaction >'

    def __call__(self):
        return self.data

    def resample(self, type_='1min'):
        """resample methods

        Returns:
            [type] -- [description]
        """

        return StockMin(tick_resample(self.data, type_))

    def get_big_orders(self, bigamount=1000000):
        """return big order

        Keyword Arguments:
            bigamount {[type]} -- [description] (default: {1000000})

        Returns:
            [type] -- [description]
        """

        return self.data.query('amount>={}'.format(bigamount))

    def get_medium_order(self, lower=200000, higher=1000000):
        """return medium

        Keyword Arguments:
            lower {[type]} -- [description] (default: {200000})
            higher {[type]} -- [description] (default: {1000000})

        Returns:
            [type] -- [description]
        """

        return self.data.query('amount>={}'.format(lower)
                               ).query('amount<={}'.format(higher))

    def get_small_order(self, smallamount=200000):
        """return small level order

        Keyword Arguments:
            smallamount {[type]} -- [description] (default: {200000})

        Returns:
            [type] -- [description]
        """

        return self.data.query('amount<={}'.format(smallamount))

    def get_time(self, start, end=None):
        if end is None:
            return self.data.loc[start]
        else:
            return self.data.loc[start:end]


class IndexTransaction():

    def __init__(self, DataFrame):
        """Index Transaction

        Arguments:
            DataFrame {pd.Dataframe} -- [input is one/multi day transaction]
        """

        self.type = 'index_transaction'

        self.data = DataFrame
        if 'amount' not in DataFrame.columns:
            if 'vol' in DataFrame.columns:
                self.data['amount'] = self.data.vol * self.data.price * 100
            elif 'volume' in DataFrame.columns:
                self.data['amount'] = self.data.volume * self.data.price * 100
        if '_id' in DataFrame.columns:
            self.data = self.data.drop(["_id"], axis=1)
        self.mongo_coll = DATABASE.index_transaction

    @property
    @lru_cache()
    def buyorsell(self):
        """return the buy or sell towards 0--buy 1--sell 2--none

        Decorators:
            lru_cache

        Returns:
            [pd.Series] -- [description]
        """

        return self.data.buyorsell

    @property
    @lru_cache()
    def price(self):
        """return the deal price of tick transaction

        Decorators:
            lru_cache

        Returns:
            [type] -- [description]
        """

        return self.data.price

    @property
    @lru_cache()
    def vol(self):
        """return the deal volume of tick

        Decorators:
            lru_cache

        Returns:
            pd.Series -- volume of transaction
        """

        try:
            return self.data.volume
        except:
            return self.data.vol

    volume = vol

    @property
    @lru_cache()
    def date(self):
        """return the date of transaction

        Decorators:
            lru_cache

        Returns:
            pd.Series -- date of transaction
        """

        return self.data.date

    @property
    @lru_cache()
    def time(self):
        """return the exact time of transaction(to minute level)

        Decorators:
            lru_cache

        Returns:
            pd.Series -- till minute level
        """

        return self.data.time

    @property
    @lru_cache()
    def datetime(self):
        """return the datetime of transaction

        Decorators:
            lru_cache

        Returns:
            pd.Series -- [description]
        """

        return self.data.datetime

    @property
    @lru_cache()
    def order(self):
        """return the order num of transaction/ for everyday change

        Decorators:
            lru_cache

        Returns:
            pd.series -- [description]
        """

        return self.data.order

    @property
    @lru_cache()
    def index(self):
        """return the transaction index

        Decorators:
            lru_cache

        Returns:
            [type] -- [description]
        """

        return self.data.index

    @property
    @lru_cache()
    def amount(self):
        """return current tick trading amount

        Decorators:
            lru_cache

        Returns:
            [type] -- [description]
        """

        return self.data.amount

    """
    最新:IF(ISNULL(NEW),PRE,NEW);
    IF (ISNULL(RANGE_AVG_PRICE) OR RANGE_AVG_PRICE <= 0)
    {
        IF (MARKETTYPE == 232 OR MARKETTYPE == 56 OR MARKETTYPE==64 OR MARKETTYPE==128 OR MARKETTYPE==168 OR MARKETTYPE==184 OR MARKETTYPE == 200 OR MARKETTYPE == 80 OR (VOL > 1 AND VOL<100))
        {
            b=SUBSAMEDAY(&VOL) ;
            m=SUM(b*最新,0);
            均价:IF(m>0,m/VOL,PRE);
        }
        ELSE IF(CODETYPE!=0 AND MONEY>0)
        {
            IF(ISNULL(MONEY) OR ISNULL(VOL) OR VOL==0 OR MONEY==0)
                均价:PRE;
            ELSE IF(VOL==VOL[1] OR MONEY==MONEY[1])
                均价:均价[1];
            ELSE
                均价:MONEY/VOL;
        }
        ELSE IF (MARKETTYPE == 176)
        {
            b=SUBSAMEDAY(&MONEY);
            m=SUM(b*最新,0);
            IF(m>0)
                均价:m/MONEY;
        }
    }
    ELSE
    {
        均价:RANGE_AVG_PRICE;
    }
    DRAWGBK(MARKETTYPE==32 AND FORMATTIME(1)<10 AND TRADETIME>242),RGB(0,0,128);
    RETURN;


    hx_star;
    hx_star_p;
    """

    def __repr__(self):
        return '< QA_DataStruct_Index_Transaction >'

    def __call__(self):
        return self.data

    def resample(self, type_='1min'):
        """resample methods

        Returns:
            [type] -- [description]
        """

        return IndexMin(tick_resample(self.data, type_))

    def get_big_orders(self, bigamount=1000000):
        """return big order

        Keyword Arguments:
            bigamount {[type]} -- [description] (default: {1000000})

        Returns:
            [type] -- [description]
        """

        return self.data.query('amount>={}'.format(bigamount))

    def get_medium_order(self, lower=200000, higher=1000000):
        """return medium

        Keyword Arguments:
            lower {[type]} -- [description] (default: {200000})
            higher {[type]} -- [description] (default: {1000000})

        Returns:
            [type] -- [description]
        """

        return self.data.query('amount>={}'.format(lower)
                               ).query('amount<={}'.format(higher))

    def get_small_order(self, smallamount=200000):
        """return small level order

        Keyword Arguments:
            smallamount {[type]} -- [description] (default: {200000})

        Returns:
            [type] -- [description]
        """

        return self.data.query('amount<={}'.format(smallamount))

    def get_time(self, start, end=None):
        if end is None:
            return self.data.loc[start]
        else:
            return self.data.loc[start:end]


class QA_DataStruct_Day(_quotation_base):
    """这个类是个通用类 一般不使用  特定生成的时候可能会用到 只具备基类方法

    Arguments:
        _quotation_base {[type]} -- [description]
    """

    def __init__(self, data, dtype='unknown_day', if_fq='bfq'):
        '''
        '''
        super().__init__(data, dtype, if_fq)


class QA_DataStruct_Min(_quotation_base):
    '''这个类是个通用类 一般不使用  特定生成的时候可能会用到 只具备基类方法
    '''

    def __init__(self, data, dtype='unknown_min', if_fq='bfq'):
        super().__init__(data, dtype, if_fq)


class _realtime_base():
    """
    realtime 基类

    主要字段有:
    code/name
    time
    open/high/low

    买卖报价队列:(不同的可能不一样 只提供list)
    ask_list[ask1_price/ask1_volume|ask2_price/ask2_volume|ask3_price/ask3_volume....]
    bid_list[bid1_price/bid1_volume|bid2_price/bid2_volume|bid3_price/bid3_volume....]
    """

    def __init__(self, market_data):
        """转化成dict模式

        Arguments:
            market_data {[type]} -- [description]
        """

        if isinstance(market_data, dict):
            self.data = market_data
        elif isinstance(market_data, pd.DataFrame):
            self.data = QA_util_to_json_from_pandas(market_data)

    @property
    def open(self):
        return self.data.get('open', None)

    @property
    def price(self):
        return self.data.get('price', None)

    @property
    def datetime(self):
        return self.data.get('datetime', None)

    @property
    def high(self):
        return self.data.get('high', None)

    @property
    def low(self):
        return self.data.get('low', None)

    @property
    def code(self):
        return self.data.get('code', None)

    @property
    def last_close(self):
        return self.data.get('last_close', None)

    @property
    def cur_vol(self):
        return self.data.get('cur_vol', None)

    @property
    def bid1(self):
        return self.data.get('bid1', None)

    @property
    def bid_vol1(self):
        return self.data.get('bid_vol1', None)

    @property
    def bid2(self):
        return self.data.get('bid2', None)

    @property
    def bid_vol2(self):
        return self.data.get('bid_vol2', None)

    @property
    def bid3(self):
        return self.data.get('bid3', None)

    @property
    def bid_vol3(self):
        return self.data.get('bid_vol3', None)

    @property
    def bid4(self):
        return self.data.get('bid4', None)

    @property
    def bid_vol4(self):
        return self.data.get('bid_vol4', None)

    @property
    def bid5(self):
        return self.data.get('bid5', None)

    @property
    def bid_vol5(self):
        return self.data.get('bid_vol5', None)

    @property
    def ask1(self):
        return self.data.get('ask1', None)

    @property
    def ask_vol1(self):
        return self.data.get('ask_vol1', None)

    @property
    def ask2(self):
        return self.data.get('ask2', None)

    @property
    def ask_vol2(self):
        return self.data.get('ask_vol2', None)

    @property
    def ask3(self):
        return self.data.get('ask3', None)

    @property
    def ask_vol3(self):
        return self.data.get('ask_vol3', None)

    @property
    def ask4(self):
        return self.data.get('ask4', None)

    @property
    def ask_vol4(self):
        return self.data.get('ask_vol4', None)

    @property
    def ask5(self):
        return self.data.get('ask5', None)

    @property
    def ask_vol5(self):
        return self.data.get('ask_vol5', None)


class QA_DataStruct_Stock_realtime(_realtime_base):

    def __init__(self, data):
        self.data = data
        self.index = data.index

    def __repr__(self):
        return '< QA_REALTIME_STRUCT >'

    # @property
    # def ask_list(self):
    #     return self.data.loc[:, ['ask1', 'ask_vol1', 'bid1', 'bid_vol1', 'ask2', 'ask_vol2',
    #                                    'bid2', 'bid_vol2', 'ask3', 'ask_vol3', 'bid3', 'bid_vol3', 'ask4',
    #                                    'ask_vol4', 'bid4', 'bid_vol4', 'ask5', 'ask_vol5', 'bid5', 'bid_vol5']]

    # @property
    # def bid_list(self):
    #     return self.data.loc[:, ['bid1', 'bid_vol1', 'bid2', 'bid_vol2',  'bid3', 'bid_vol3', 'bid4', 'bid_vol4', 'bid5', 'bid_vol5']]

    @property
    def _data(self):
        """
        return a dataframe-type result
        """
        return pd.DataFrame(self.data)

    @property
    def datetime(self):
        return self.index.levels[0]

    @property
    def code(self):
        return self.index.levels[1]

    def serialize(self):
        """to_protobuf
        """
        pass

    def to_json(self):
        return self.data.assign(
            code=self.code,
            datetime=str(self.datetime)
        ).to_dict(orient='records')

    def resample(self, level):
        return tick_resample(self.data, level)


QA_DataStruct_Future_realtime = QA_DataStruct_Stock_realtime


class QA_DataStruct_Stock_realtime_series():

    def __init__(self, sr_series):

        if isinstance(sr_series[0], QA_DataStruct_Stock_realtime):
            self.sr_series = sr_series
        elif isinstance(sr_series[0], dict):
            self.sr_series = [
                QA_DataStruct_Stock_realtime(sr) for sr in sr_series
            ]
        self.table = pd.concat([sr._data for sr in self.sr_series])


class QA_DataStruct_Security_list():

    def __init__(self, DataFrame):
        self.data = DataFrame.loc[:,
                    ['sse',
                     'code',
                     'name']].set_index(
            'code',
            drop=False
        )

    @property
    def code(self):
        return self.data.code

    @property
    def name(self):
        return self.data.name

    def get_stock(self, ST_option):
        return self.data

    def get_index(self):
        return self.data

    def get_etf(self):
        return self.data


class QA_DataStruct_Future_tick():
    """
    CTP FORMAT
    {'TradingDay': '20181115',
    'InstrumentID': 'rb1901',
    'ExchangeID': '',
    'ExchangeInstID': '',
    'LastPrice': 3874.0,
    'PreSettlementPrice': 3897.0,
    'PreClosePrice': 3937.0,
    'PreOpenInterest': 2429820.0,
    'OpenPrice': 3941.0,
    'HighestPrice': 3946.0,
    'LowestPrice': 3865.0,
    'Volume': 2286142,
    'Turnover': 89450228460.0,
    'OpenInterest': 2482106.0,
    'ClosePrice': 1.7976931348623157e+308,
    'SettlementPrice': 1.7976931348623157e+308,
    'UpperLimitPrice': 4169.0,
    'LowerLimitPrice': 3624.0,
    'PreDelta': 0.0,
    'CurrDelta': 1.7976931348623157e+308,
    'BidPrice1': 3873.0,
    'BidVolume1': 292,
    'AskPrice1': 3874.0,
    'AskVolume1': 223,
    'BidPrice2': 1.7976931348623157e+308,
    'BidVolume2': 0,
    'AskPrice2': 1.7976931348623157e+308,
    'AskVolume2': 0,
    'BidPrice3': 1.7976931348623157e+308,
    'BidVolume3': 0,
    'AskPrice3': 1.7976931348623157e+308,
    'AskVolume3': 0,
    'BidPrice4': 1.7976931348623157e+308,
    'BidVolume4': 0,
    'AskPrice4': 1.7976931348623157e+308,
    'AskVolume4': 0,
    'BidPrice5': 1.7976931348623157e+308,
    'BidVolume5': 0,
    'AskPrice5': 1.7976931348623157e+308,
    'AskVolume5': 0,
    'AveragePrice': 39127.15328269198,
    'ActionDay': '20181115'
    'UpdateTime': '11:30:01',
    'UpdateMillisec': 0,}

    replace(1.7976931348623157e+308, np.nan)
    """

    def __init__(self, data={}):
        self.data = data

    def trading_day(self, ):
        pass

    def append(self, new_data):
        pass


class StockBlock():

    def __init__(self, DataFrame):
        self.data = DataFrame
        assert isinstance(DataFrame.index, pd.MultiIndex)
        self.index = self.data.index.remove_unused_levels()

    def __repr__(self):
        return '< QA_DataStruct_Stock_Block >'

    def __call__(self):
        """调用直接返回内部的数据

        Returns:
            [type] -- [description]
        """

        return self.data

    def new(self, data):
        """通过data新建一个stock_block

        Arguments:
            data {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        temp = copy(self)
        temp.__init__(data)
        return temp

    @property
    def len(self):
        """返回DataStruct的长度

        Returns:
            [type] -- [description]
        """

        return len(self.data)

    @property
    def block_name(self):
        """返回所有的板块名

        Returns:
            [type] -- [description]
        """

        return self.index.levels[0].tolist()

    @property
    def code(self):
        """返回唯一的证券代码

        Returns:
            [type] -- [description]
        """

        return self.index.levels[1].tolist()

    @property
    def view_code(self):
        """按股票排列的查看blockname的视图

        Returns:
            [type] -- [description]
        """

        return self.data.groupby(level=1).apply(
            lambda x:
            [item for item in x.index.remove_unused_levels().levels[0]]
        )

    @property
    def view_block(self):
        """按版块排列查看的code的视图

        Returns:
            [type] -- [description]
        """

        return self.data.groupby(level=0).apply(
            lambda x:
            [item for item in x.index.remove_unused_levels().levels[1]]
        )

    def show(self):
        """展示DataStruct

        Returns:
            dataframe -- [description]
        """

        return self.data

    def get_code(self, code):
        """getcode 获取某一只股票的板块

        Arguments:
            code {str} -- 股票代码

        Returns:
            DataStruct -- [description]
        """
        # code= [code] if isinstance(code,str) else
        return self.new(self.data.loc[(slice(None), code), :])

    def get_block(self, block_name):
        """getblock 获取板块, block_name是list或者是单个str

        Arguments:
            block_name {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        # block_name = [block_name] if isinstance(
        #     block_name, str) else block_name
        # return QA_DataStruct_Stock_block(self.data[self.data.blockname.apply(lambda x: x in block_name)])

        return self.new(self.data.loc[(block_name, slice(None)), :])

    def get_both_code(self, code):
        """get_both_code 获取几个股票相同的版块

        Arguments:
            code {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        return self.new(self.data.loc[(slice(None), code), :])

    def get_both_block(self, block_list):
        n = len(block_list)
        count = self.get_block(block_list).data.assign(countx=1).groupby('code').countx.count()
        return count[count == n].index.tolist()

    # def getdtype(self, dtype):
    #     """getdtype

    #     Arguments:
    #         dtype {str} -- gn-概念/dy-地域/fg-风格/zs-指数

    #     Returns:
    #         [type] -- [description]
    #     """

    #     return QA_DataStruct_Stock_block(self.data[self.data['type'] == dtype])

    # def get_price(self, _block_name=None):
    #     """get_price

    #     Keyword Arguments:
    #         _block_name {[type]} -- [description] (default: {None})

    #     Returns:
    #         [type] -- [description]
    #     """

    #     if _block_name is not None:
    #         try:
    #             code = self.data[self.data['blockname']
    #                              == _block_name].code.unique().tolist()
    #             # try to get a datastruct package of lastest price
    #             return QA_fetch_get_stock_realtime(code)

    #         except:
    #             return "Wrong Block Name! Please Check"
    #     else:
    #         code = self.data.code.unique().tolist()
    #         return QA_fetch_get_stock_realtime(code)


class Financial():

    def __init__(self, data):
        self.data = data
        # keys for CN, values for EN
        self.colunms_en = list(financial_dict.values())
        self.colunms_cn = list(financial_dict.keys())

    def __repr__(self):
        return '< QA_DataStruct_Financial >'

    def get_report_by_date(self, code, date):
        return self.data.loc[pd.Timestamp(date), code]

    def get_key(self, code, reportdate, key):
        if isinstance(reportdate, list):
            return self.data.loc[(
                                     slice(
                                         pd.Timestamp(reportdate[0]),
                                         pd.Timestamp(reportdate[-1])
                                     ),
                                     code
                                 ),
                                 key]
        else:
            return self.data.loc[(pd.Timestamp(reportdate), code), key]
