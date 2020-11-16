import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
import os
import QUANTAXIS as qa
import datetime as dt
from attention import attention
from tfrecord import np_to_tfrecords
from index import *
from joblib import Parallel, delayed
import sys

start_date = dt.date(2015, 1, 1)
end_date = dt.date(2018, 6, 5)

# TRAINING_STEPS = 5000                        # 训练轮数。
BATCH_SIZE = 128                             # batch大小。
EPOCH_NUM = 100

pvars = ['open', 'close', 'high', 'low']

INTERVAL = 240 // 5                        # 每天打点的数量
ATTENTION_SIZE = 64
FUTURE_DAYS = 10

n_input = INTERVAL + 1
n_classes = 2

HIDDEN_SIZE = n_input    # 128                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                               # LSTM的层数。
TIMESTEPS = 60                               # 循环神经网络的训练序列长度。


def prepare_label_with_relative_position(stocks):
    print('prepare label')
    data = []
    for stock in stocks:
        try:
            candles = qa.QA_fetch_stock_day_adv(stock, start=str(start_date), end=str(end_date)).to_qfq().data
            candles['ret'] = candles.close.shift(-FUTURE_DAYS) / candles.close - 1
            data.append(candles.loc[:, ['close', 'ret']].fillna(0).reset_index())
        except:
            continue
    full = pd.concat(data)
    full['rnk'] = full.groupby('date').transform(lambda x: x.rank())['ret']
    full['label'] = (full.rnk >= (len(data) * 0.5 + 1)) * 1
    full.loc[:, 'date'] = full['date'].apply(lambda x: str(x)[:10])
    print('label prepared:\n{}'.format(full.label.value_counts()))
    return full.loc[:, ['date', 'code', 'label']]


def generate_singe_data(stock, labels, valid_date, test_date):
    print('handling %s' % stock)
    try:
        candles = qa.QA_fetch_stock_min_adv(stock, start=str(start_date), end=str(end_date), frequence='5min').to_qfq().data
    except:
        print('data error: {}'.format(stock))
        return None, None, None

    days = []
    valid_period = 0
    test_period = 0
    dates = []
    for x in range(candles.shape[0] // INTERVAL):
        day = candles.ix[x*INTERVAL: (x+1)*INTERVAL, ['open', 'close']]
        days.append(np.insert(day.close.values, 0, day.open[0]))
        valid_period += 1 if candles.ix[x*INTERVAL, 'date'] >= valid_date else 0
        test_period += 1 if candles.ix[x*INTERVAL, 'date'] >= test_date else 0
        dates.append(candles.ix[x*INTERVAL, 'date'][:10])
    cdata = pd.DataFrame(days)
    if cdata.shape[0] < 200:  # if trade days are less than XXX, ignore the stock
        print('trade days too short: {}'.format(stock))
        return None, None, None

    price_changes = []
    price_dates = []
    for x in range(cdata.shape[0]-TIMESTEPS+1):
        today = cdata.iloc[x: x+TIMESTEPS, :] / cdata.iloc[x+TIMESTEPS-1, -1] - 1
        price_changes.append(today.values.flatten())
        price_dates.append(dates[x+TIMESTEPS-1])

    if len(price_changes) > valid_period:
        train = pd.DataFrame(price_changes[:-valid_period])
        train['code'] = stock
        train['date'] = price_dates[:-valid_period]
        train = pd.merge(train, labels, on=['date', 'code'])
    else:
        train = None

    if len(price_changes) > test_period and valid_period > test_period:
        valid = pd.DataFrame(price_changes[-valid_period:-test_period])
        valid['code'] = stock
        valid['date'] = price_dates[-valid_period:-test_period]
        valid = pd.merge(valid, labels, on=['date', 'code'])
    else:
        valid = None

    if test_period > 0:
        test = pd.DataFrame(price_changes[-test_period:])
        test['code'] = stock
        test['date'] = price_dates[-test_period:]
    else:
        test = None
    return train, valid, test


def generate_data(stocks, valid_date, test_date):
    # use 沪指==000001, possible alternative is 中证800=='000906'

    labels = prepare_label_with_relative_position(stocks)

    data = Parallel(n_jobs=4)(delayed(generate_singe_data)(stock, labels, valid_date, test_date) for stock in stocks)
    train_data = [x[0] for x in data if x[0] is not None]
    valid_data = [x[1] for x in data if x[1] is not None]
    test_data = [x[2] for x in data if x[1] is not None]

    return pd.concat(train_data), pd.concat(valid_data), pd.concat(test_data)


def lstm_model(X, y, is_training):

    # 规整成矩阵数据
    X = tf.reshape(X, [-1, TIMESTEPS, n_input])

    ## 规整输入的数据
    ## 输入层到隐含层，第一次是直接运算
    # X = tf.transpose(X, [1, 0, 2])  # permute n_steps and batch_size
    # X = tf.reshape(X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # X = tf.matmul(X, weights['hidden']) + biases['hidden']  # tf.matmul(a,b)   将矩阵a乘以矩阵b，生成a * b

    # 之后使用LSTM

    # 使用多层的LSTM结构。
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    # X = tf.split(X, TIMESTEPS, 0)
    X = tf.reshape(X, [-1, TIMESTEPS, HIDDEN_SIZE])
    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE),
    #                                              tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), X, dtype=tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # output = outputs[:, -1, :]
    with tf.name_scope('Attention_layer'):
        attention_output = attention(outputs, (w_omega, b_omega, u_omega), return_alphas=False)

    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    after_attention = tf.nn.relu(tf.matmul(attention_output, weights['att']) + biases['att'])
    predictions = tf.matmul(after_attention, weights['out']) + biases['out']

    prob = predictions[:, 1]
    # predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return prob, None, None

    # 计算损失函数。
    # y = tf.reshape(y, [-1, 1])
    # loss = tf.reduce_sum(tf.sqrt(tf.multiply(tf.squared_difference(y, predictions),
    #                                          tf.cast(tf.logical_and(tf.less(y, tf.zeros_like(y) - 0.01),
    #                                                                 tf.greater(predictions, tf.zeros_like(y) + 0.01)), tf.float32) * 4 +
    #                                          #                                 #tf.cast(tf.less(y, predictions), tf.float32) * 1 +
    #                                          tf.ones_like(y))))
    # loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    # y = tf.reshape(y, [BATCH_SIZE, 1])
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=y))

    # pred = tf.argmax(predictions, 1)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adam", learning_rate=0.001)
    return prob, loss, train_op


def run_day_eval(sess, test):
    count = test.groupby('date').count().code
    valid_date = count[count > 100].index.tolist()
    hit_rate = []
    for date in valid_date:
        print('evaluate %s' % date)
        test_data = test.loc[test.date == date, :]
        test_X = test_data.iloc[:, :-3].values
        test_y = test_data.iloc[:, -1].values
        codes = test_data.loc[:, 'code'].tolist()

        ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        ds = ds.batch(1)
        X, y = ds.make_one_shot_iterator().get_next()

        # 调用模型得到计算结果。这里不需要输入真实的y值。
        with tf.variable_scope("eval", reuse=True):
            prediction, _, _ = lstm_model(X, [0.0], False)

        # 将预测结果存入一个数组。
        predictions = []
        hit = 0.
        for i in range(len(test_X)):
            pred, l = sess.run([prediction, y])
            predictions.append({codes[i], pred, l})
        predictions.sort(key=lambda x: x[1], reverse=True)
        for p in predictions[:10]:
            print('{}: predicted {}, real {}'.format(p[0], p[1], p[2]))
            if p[2] > 0:
                hit += 1
        hit_rate.append(hit / 10.)
        print('\n-------------\n')
    print('hit rate: {}'.format(np.array(hit_rate).mean()))


def run_valid(sess, valid):
    valid_X = valid.drop(columns=['date', 'code', 'label']).values.astype(np.float32)
    valid_y = valid.loc[:, 'label'].values.astype(np.int32)

    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((valid_X, valid_y)).batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("valid", reuse=True):
        prediction, _, _ = lstm_model(X, None, False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(len(valid_X)):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    valid['predict'] = predictions
    valid.loc[:, ['date', 'code', 'label', 'predict']].to_csv('../data/rnn_rel_valid_predict.csv', index=False)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)


def run_test(sess, test):
    test_X = test.drop(columns=['date', 'code']).values.astype(np.float32)

    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices(test_X).batch(1)
    X = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("test", reuse=True):
        prediction, _, _ = lstm_model(X, None, False)

    # 将预测结果存入一个数组。
    predictions = []
    for i in range(len(test_X)):
        p = sess.run(prediction)
        predictions.append(p)

    test['predict'] = predictions
    test.loc[:, ['date', 'code', 'predict']].sort_values(['date', 'predict'], ascending=False).to_csv('../data/rnn_rel_test_predict.csv', index=False)


def parser(record):
    features = {
        'X': tf.FixedLenFeature([(INTERVAL+1)*TIMESTEPS], tf.float32),
        'Y': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, features)

    # Perform additional preprocessing on the parsed data.
    label = tf.cast(parsed["Y"], tf.int32)
    # label = tf.stack([label, -(label-1)], axis=-1)
    return parsed['X'], label


if __name__ == '__main__':

    # print(len(data))
    # print(data[:2])
    # print(label[:2])
    stocks = [x for x in ZZ800.split('\n') if len(x) > 0]

    if os.path.exists('../data/rnn_rel_valid.hdf'):
        #train = pd.read_csv('../data/rnn_rel_train.csv', dtype={'code': np.object})
        valid = pd.read_hdf('../data/rnn_rel_valid.hdf', 'data')
        test = pd.read_hdf('../data/rnn_rel_test.hdf', 'data')
        print('data read')

    else:
        train, valid, test = generate_data(stocks, '2018-02-01', '2018-05-01')

        train_X = train.drop(columns=['date', 'code', 'label']).values.astype(np.float32)
        train_y = train.loc[:, 'label'].values.astype(np.int64)
        train_y = train_y.reshape([train_y.shape[0], 1])

        np_to_tfrecords(train_X, train_y, '../data/rnn_rel_train')
        # train.to_csv('../data/rnn_rel_train.csv', index=False)
        valid.to_hdf('../data/rnn_rel_valid.hdf', 'data')
        test.to_hdf('../data/rnn_rel_test.hdf', 'data')
        print('data saved')

    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, HIDDEN_SIZE], stddev=0.1)),  # Hidden layer weights
        'att': tf.Variable(tf.truncated_normal([HIDDEN_SIZE, 128], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([128, n_classes], stddev=0.1))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1)),
        'att': tf.Variable(tf.truncated_normal([128], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_classes], stddev=0.1))
    }

    w_omega = tf.Variable(tf.random_normal([HIDDEN_SIZE, ATTENTION_SIZE], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([ATTENTION_SIZE], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([ATTENTION_SIZE], stddev=0.1))

    # 将训练数据以数据集的方式提供给计算图。
    tds = tf.data.TFRecordDataset('../data/rnn_rel_train.tfrecords')
    tds = tds.map(parser)
    # tds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    tds = tds.repeat(EPOCH_NUM).shuffle(12800).batch(BATCH_SIZE)
    t_X, t_y = tds.make_one_shot_iterator().get_next()

    # 定义模型，得到预测结果、损失函数，和训练操作。
    with tf.variable_scope("model"):
        _, loss, train_op = lstm_model(t_X, t_y, True)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 测试在训练之前的模型效果。
        # print("Evaluate model before training.")
        # run_eval(sess, test_X, test_y)

        # print(sess.run(x_shape))

        # 训练模型。
        step = 0
        # for i in range(TRAINING_STEPS):
        while True:
            try:
                # if step == 0:
                #     data_x, data_y = sess.run([t_X, t_y])
                #     print('data:\n{}\nlabel:\n{}'.format(data_x, data_y))
                _, l = sess.run([train_op, loss])
                if step % 1000 == 0:
                    print("train step: " + str(step) + ", loss: " + str(l))
                    saver.save(sess, '../model/rnn0606', global_step=step)
            except tf.errors.OutOfRangeError as e:
                break
            step += 1

        # 使用训练好的模型对测试数据进行预测。
        print("Evaluate model after training.")
        run_valid(sess, valid)

        print('Predict test values')
        run_day_eval(sess, test)
