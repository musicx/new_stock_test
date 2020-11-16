import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from attention import attention
import QUANTAXIS as qa

n_input = 5
n_classes = 2

HIDDEN_SIZE = n_input  # 32                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # LSTM的层数。
TIMESTEPS = 30                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 5000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
EPOCH_NUM = 100

pvars = ['open', 'close', 'high', 'low']

single_stock = '002635'

INDEX = True
ATTENTION_SIZE = 32
MIN = True
TEST_PERIOD = 30


def generate_data(stocks):
    first = True
    train_X = np.array([])
    train_y = np.array([])
    test_X = np.array([])
    test_y = np.array([])
    base = qa.QA_fetch_index_day_adv('000001', start='2010-01-01', end='2018-06-02').data
    for stock in stocks:
        candles = qa.QA_fetch_stock_day_adv(stock, start='2010-01-01', end='2018-06-02').data
        cdata = candles.loc[:, pvars]
        cdata.loc[:, 'base'] = base.close / base.close.shift(1)
        cdata.loc[:, 'ret'] = cdata.close / cdata.close.shift(1)
        cdata.loc[:, 'relative'] = cdata.ret - cdata.base
        cdata.fillna(0, inplace=True)

        pdata = []
        plabel = []
        for x in range(cdata.shape[0]-TIMESTEPS):
            day = cdata.ix[x: x+TIMESTEPS, pvars] / cdata.ix[x+TIMESTEPS-1, 'close'] - 1
            day['base'] = cdata.ix[x: x+TIMESTEPS, 'relative']
            pdata.append(day.values.flatten())
            # ret = cdata.ix[x+TIMESTEPS, 'close'] / cdata.ix[x+TIMESTEPS-1, 'close'] - 1
            plabel.append(1 if cdata.ix[x+TIMESTEPS, 'relative'] > 0 else 0)
        if len(pdata) > TEST_PERIOD:
            data = np.asarray(pdata, dtype=np.float32)
            label = np.asarray(plabel, dtype=np.int32)
            if first:
                train_X = data[:-TEST_PERIOD]
                train_y = label[:-TEST_PERIOD]
                test_X = data[-TEST_PERIOD:]
                test_y = label[-TEST_PERIOD:]
                first = False
            else:
                train_X = np.r_[train_X, data[:-TEST_PERIOD]]
                train_y = np.r_[train_y, label[:-TEST_PERIOD]]
                test_X = np.r_[test_X, data[-TEST_PERIOD:]]
                test_y = np.r_[test_y, label[-TEST_PERIOD:]]

    # np.save('../data/idata_000001', data)
    # np.save('../data/ilabel_000001', label)
    # print('data saved\n{}'.format(label))
    return train_X, train_y, test_X, test_y


train_X, train_y, test_X, test_y = generate_data(['000001', '600000'])


# train_y = np.array([train_y, -(train_y - 1)]).T   # need this ?
# test_y = np.array([test_y, -(test_y - 1)]).T   # need this ?
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


def lstm_model(X, y, is_training):

    # 规整成矩阵数据
    X = tf.reshape(X, [-1, TIMESTEPS, n_input])

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

    # prob = predictions[:, 1]
    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return predictions, None, None

    # 计算损失函数。
    # y = tf.reshape(y, [-1, 1])
    # loss = tf.reduce_sum(tf.sqrt(tf.multiply(tf.squared_difference(y, predictions),
    #                                          tf.cast(tf.logical_and(tf.less(y, tf.zeros_like(y) - 0.001),
    #                                                                 tf.greater(predictions, tf.zeros_like(y) + 0.001)), tf.float32) * 4 +
    # #                                 #tf.cast(tf.less(y, predictions), tf.float32) * 1 +
    #                              tf.ones_like(y))))
    # loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predictions))

    # pred = tf.argmax(predictions, 1)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adam", learning_rate=0.001)
    return predictions, loss, train_op


def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, None, False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(len(test_X)):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)
        print('label: {}, predicted: {}'.format(l, p))

    # 计算rmse作为评价指标。
    # predictions = np.array(predictions).squeeze()
    # labels = np.array(labels).squeeze()
    # rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    # print("Root Mean Square Error is: %f" % rmse)
    #
    # # 对预测的sin函数曲线进行绘图。
    # plt.figure()
    # plt.plot(predictions * 10, label='predictions')
    # plt.plot(labels, label='real_close')
    # plt.legend()
    # plt.show()


# 将训练数据以数据集的方式提供给计算图。
tds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
tds = tds.repeat(EPOCH_NUM).shuffle(1000).batch(BATCH_SIZE)
t_X, t_y = tds.make_one_shot_iterator().get_next()


# 定义模型，得到预测结果、损失函数，和训练操作。
with tf.variable_scope("model"):
    _, loss, train_op = lstm_model(t_X, t_y, True)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 测试在训练之前的模型效果。
    print("\nEvaluate model before training.\n")
    run_eval(sess, test_X, test_y)

    # print(sess.run(x_shape))

    # 训练模型。
    step = 0
    # for i in range(TRAINING_STEPS):
    while True:
        try:
            _, l = sess.run([train_op, loss])
            if step % 100 == 0:
                print("train step: " + str(step) + ", loss: " + str(l))
                # saver.save(sess, '../model/rnn0508', global_step=step)
        except tf.errors.OutOfRangeError as e:
            break
        step += 1

    saver.save(sess, '../model/rnn0508')
    # 使用训练好的模型对测试数据进行预测。
    print("\n\nEvaluate model after training.\n")
    run_eval(sess, test_X, test_y)
