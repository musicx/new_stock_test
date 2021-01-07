import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

def list_division(list1, list2):
    u'''need：(list1,list2)  return：list return.list.len：A
    列表除：两个列表相同长度时,相同位置相除；
    若其中一个列表为常数,则用另一列表的每个元素除(被除)此常数。'''
    if type(list1) == float: return[list1/y for y in list2]
    if type(list2) == float: return[x/list2 for x in list1]
    return[x/y for x, y in zip(list1, list2)]


def list_subtraction(list1, list2):
    u'''need：(list1,list2)  return：list return.list.len：A
    列表减：两个列表相同长度时,相同位置相减；
    若其中一个列表为常数,则用另一列表的每个元素减(被减)此常数。'''
    if type(list1) == float: return[list1-y for y in list2]
    if type(list2) == float: return[x-list2 for x in list1]
    return[x-y for x, y in zip(list1, list2)]


def RANK(list1):
    u'''need：(list1) return：list return.list.len：A
    向量 A 升序排序'''
    return sorted(list1)


# def MAX(number1, number2)：
    # u'''need：(number1,number2) return：number 
    # 在 A,B 中选择最大的数'''
    # return max(number1, number2)


# def MIN(number1, number2)：
    # u'''need：(number1,number2) return：number 
    # 在 A,B 中选择最小的数'''
    # return min(number1, number2)


def STD(list1, n):
    u'''need：(list,number)  return：number
    序列 list 过去 n 天的标准差'''
    mean = MEAN(list1, n)
    return sum([(x-mean)**2 for x in list1])/n**0.5


def CORR(list1, list2, n):
    u'''need：(list1,list2,number)  return：number
    序列 A、B 过去 n 天相关系数'''
    list1, list2 = list1[-n:], list2[-n:]
    corrnum = np.corrcoef(np.array([list1, list2]))[0][1]
    if np.isnan(corrnum) == False: return corrnum
    else: return 0


def DELTA(list1, n):
    u'''need：(list,number)  return：list  return.list.len：A-n
    序列 A 中每个数和第前n 天的差'''
    return [list1[x+n]-list1[x]for x in range(0, len(list1)-n)]


def LOG(list1):
    u'''need：(list1) return：list return.list.len：A
    自然对数函数'''
    return np.log(list1)


def SUM(list1, n):
    u'''need：(list,number)  return：number
    序列 list 过去 n 天求和'''
    return sum(list1[-n:])


def ABS(list1):
    u'''need：(list1) return：list return.list.len：A
    列表绝对值：分别求列表内每个元素的绝对值'''
    return[abs(x) for x in list1]


def MEAN(list1, n):
    u'''need：(list,number)  return：number
    序列 list 过去 n 天均值'''
    return sum(list1[-n:])/n


def TSRANK(list1, n):
    u'''need：(list,number)  return：number
    序列 A 的末位值在过去 n 天的顺序排位'''
    return sorted(list1[-n:]).index(list1[-1])+1


def SIGN(number):
    u'''need：(number)  return：number
    符号函数'''
    if number > 0: return 1
    elif number == 0: return 0
    elif number < 0: return -1


def COVIANCE(list1, list2, n):
    u'''need：(list,number,number)  return：number
    序列 A、B 过去 n 天协方差'''
    covnum=np.cov([list1,list2])[0][1]
    if np.isnan(covnum) == False: return covnum
    else: return 0


def DELAY(list1, n):
    u'''need：(list,number)  return：number
    序列 A过去 n 天的数据'''
    return list1[-n-1]


def TSMIN(list1, n):
    u'''need：(list,number)  return：number
    序列 A 过去 n 天的最小值'''
    return min(list(list1)[-n:])


def TSMAX(list1, n):
    u'''need：(list,number)  return：number
    序列 A 过去 n 天的最大值'''
    return max(list(list1)[-n:])


def PROD(list1, n):
    u'''need：(list,number)  return：number
    序列 A 过去 n 天的累乘'''
    return np.cumprod(list1[-n:])[-1]


def REGBETA(list1, list2, n):
    u'''need：(list1,list2,number)  return：number
    前 n 期样本 A 对 B 做回归所得回归系数'''
    zzz = pd.DataFrame({'list1': list1[-n:], 'list2': list2[-n:]})
    X_train, X_test, y_train, y_test = train_test_split(zzz[['list1']], zzz['list2'], random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)
    return float(linreg.coef_)


def REGRESI(list1, list2, n):
    u'''need：(list1,list2,number)  return：number
    前 n 期样本 A 对 B 做回归所得的残差'''
    zzz = pd.DataFrame({'list1': list1[-n:], 'list2': list2[-n:]})
    X_train, X_test, y_train, y_test = train_test_split(zzz[['list1']], zzz['list2'], random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)
    return float(linreg.intercept_)


def SMA(list1, n, m):
    u'''need：(list,number,number)  return：number
    SMA(A,n,m)'''
    y = [list1[0]]
    for x in range(0, len(list1)):
        y.append((list1[x]*m+y[-1]*(n-m))/n)
    return y[-1]


def WMA(list1,n):
    u'''need：(list,number)  return：number
    计算A前n期样本加权平均值'''
    a = [0.9**i for i in range(1, n+1)]
    return sum([a[i]*list1[-i-1] for i in range(0, len(a))])/sum(a)



def DECAYLINEAR(list1, d):
    u'''need：(list,number)  return：list return.list.len：A-d+1
    对 A 序列计算移动平均加权'''
    a = list(range(1, d+1))
    return[sum([a[i]*list1[i+x] for i in range(0, d)])/sum(a) for x in range(0, len(list1)-d+1)]



def HIGHDAY(list1, n):
    u'''need：(list,number)  return：number
    计算 A 前 n 期时间序列中最大值距离当前时点的间隔'''
    return n-list1[-n:].index(TSMAX(list1, n))-1


def LOWDAY(list1, n):
    u'''need：(list,number)  return：number
    计算 A 前 n 期时间序列中最小值距离当前时点的间隔'''
    return n-list1[-n:].index(TSMIN(list1, n))-1



def SEQUENCE(n):
    u'''need：(number)  return：list return.list.len：n
    生成 1~n 的等差序列'''
    return list(range(1, n+1))


def SUMAC(list1, n):
    u'''need：(list,number)  return：number
    计算 A 的前 n 项的累加'''
    return sum(list1[:n])