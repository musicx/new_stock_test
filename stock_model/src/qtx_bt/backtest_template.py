
from QUANTAXIS.QAARP.QARisk import QA_Risk
from QUANTAXIS.QAARP.QAUser import QA_User
from QUANTAXIS.QAApplication.QABacktest import QA_Backtest
from QUANTAXIS.QAARP.QAStrategy import QA_Strategy
import threading
from QUANTAXIS.QAARP.QAAccount import QA_Account
from QUANTAXIS.QAUtil.QALogs import QA_util_log_info
from QUANTAXIS.QAUtil.QAParameter import (AMOUNT_MODEL, FREQUENCE, MARKET_TYPE,
                                          ORDER_DIRECTION, ORDER_MODEL)





def run_daybacktest():
    import QUANTAXIS as QA
    backtest = Backtest(market_type=MARKET_TYPE.STOCK_CN,
                        frequence=FREQUENCE.DAY,
                        start='2017-01-01',
                        end='2017-02-10',
                        code_list=QA.QA_fetch_stock_block_adv().code[0:5],
                        commission_fee=0.00015)
    print(backtest.account)
    backtest.start_market()

    backtest.run()
    backtest.stop()



if __name__ == '__main__':
    run_daybacktest()
    #run_minbacktest()