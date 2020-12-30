# coding=utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2020 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from QUANTAXIS.QAARP.QARisk import QA_Risk
from QUANTAXIS.QAARP.QAUser import QA_User
from QUANTAXIS.QAApplication.QABacktest import QA_Backtest
from QUANTAXIS.QAARP.QAStrategy import QA_Strategy
import threading
from QUANTAXIS.QAARP.QAAccount import QA_Account
from QUANTAXIS.QAUtil.QALogs import QA_util_log_info
from QUANTAXIS.QAUtil.QAParameter import (AMOUNT_MODEL, FREQUENCE, MARKET_TYPE,
                                          ORDER_DIRECTION, ORDER_MODEL)

import random

class MAStrategy(QA_Account):
    def __init__(self, user_cookie, portfolio_cookie, account_cookie,  init_cash=100000, init_hold={}):
        super().__init__(user_cookie=user_cookie, portfolio_cookie=portfolio_cookie, account_cookie= account_cookie,
                         init_cash=init_cash, init_hold=init_hold)
        self.frequence = FREQUENCE.DAY
        self.market_type = MARKET_TYPE.STOCK_CN
        self.commission_coeff = 0.00015
        self.tax_coeff = 0.0001
        self.reset_assets(100000)  # 这是第二种修改办法

    def on_bar(self, event):
        print(threading.enumerate())
        sellavailable = self.sell_available
        try:
            for item in event.market_data.code:
                if sellavailable.get(item, 0) > 0:
                    event.send_order(account_cookie=self.account_cookie,
                                     amount=sellavailable[item], amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=0,
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.SELL,
                                     market_type=self.market_type, frequence=self.frequence,
                                     broker_name=self.broker
                                     )
                else:
                    event.send_order(account_cookie=self.account_cookie,
                                     amount=100, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=0,
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                     market_type=self.market_type, frequence=self.frequence,
                                     broker_name=self.broker)

        except Exception as e:
            print(e)


class MAMINStrategy(QA_Strategy):
    def __init__(self):
        super().__init__()
        self.frequence = FREQUENCE.FIFTEEN_MIN
        self.market_type = MARKET_TYPE.STOCK_CN

    def on_bar(self, event):
        try:
            # 新数据推送进来
            for item in event.market_data.code:
                # 如果持仓
                if self.sell_available.get(item, 0) > 0:
                    # 全部卖出
                    event.send_order(account_cookie=self.account_cookie,
                                     amount=self.sell_available[item], amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=0,
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.SELL,
                                     market_type=self.market_type, frequence=self.frequence,
                                     broker_name=self.broker
                                     )
                else:  # 如果不持仓
                    # 买入1000股
                    event.send_order(account_cookie=self.account_cookie,
                                     amount=100, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=0,
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                     market_type=self.market_type, frequence=self.frequence,
                                     broker_name=self.broker)
        except:
            pass


class Backtest(QA_Backtest):
    '''
    多线程模式回测示例

    '''

    def __init__(self, market_type, frequence, start, end, code_list, commission_fee):
        super().__init__(market_type,  frequence, start, end, code_list, commission_fee)
        mastrategy = MAStrategy(user_cookie=self.user.user_cookie, portfolio_cookie= self.portfolio.portfolio_cookie, account_cookie= 'mastrategy')
        #maminstrategy = MAMINStrategy()
        self.account = self.portfolio.add_account(mastrategy)

    def after_success(self):
        QA_util_log_info(self.account.history_table)
        risk = QA_Risk(self.account, benchmark_code='000300',
                       benchmark_type=MARKET_TYPE.INDEX_CN)

        print(risk().T)
        fig=risk.plot_assets_curve()
        fig.show()
        fig=risk.plot_dailyhold()
        fig.show()
        fig=risk.plot_signal()
        fig.show()
        self.account.save()
        risk.save()


def run_daybacktest():
    import QUANTAXIS as QA
    backtest = Backtest(market_type=MARKET_TYPE.STOCK_CN,
                        frequence=FREQUENCE.DAY,
                        start='2017-01-01',
                        end='2017-02-10',
                        code_list=QA.QA_fetch_stock_block_adv().code[0:5],
                        commission_fee=0.00015)
    backtest._generate_account()
    print(backtest.account)
    backtest.start_market()

    backtest.run()
    backtest.stop()


def run_minbacktest():
    import QUANTAXIS as QA
    backtest = Backtest(market_type=MARKET_TYPE.STOCK_CN,
                        frequence=FREQUENCE.FIFTEEN_MIN,
                        start='2017-11-01',
                        end='2017-11-10',
                        code_list=QA.QA_fetch_stock_block_adv().code[0:5],
                        commission_fee=0.00015)
    backtest.start_market()

    backtest.run()
    backtest.stop()


if __name__ == '__main__':
    random.seed(10)
    run_daybacktest()
    #run_minbacktest()

