import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from attention import attention

HIDDEN_SIZE = 32                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 1                              # LSTM的层数。
TIMESTEPS = 30                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 5000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
EPOCH_NUM = 100

pvars = ['open', 'close', 'high', 'low']

single_stock = '002635'

INDEX = True
INTERVAL = 240 // 5
ATTENTION_SIZE = 32
MIN = True
TEST_PERIOD = 30


def generate_data(stock):
    if os.path.exists("idata_000001.npy"):
        data = np.load('idata_000001.npy')
        label = np.load('ilabel_000001.npy')
    elif INDEX:
        import QUANTAXIS as qa
        candles = qa.QA_fetch_index_min('000001', start='2015-01-01', end='2018-05-08',
                                        format='pandas', frequence='5min')
        days = []
        for x in range(candles.shape[0] // INTERVAL):
            day = candles.ix[x*INTERVAL : (x+1)*INTERVAL, ['open', 'close']]
            days.append(np.insert(day.close.values, 0, day.open[0]))
        cdata = pd.DataFrame(days)

        pdata = []
        plabel = []
        for x in range(cdata.shape[0]-TIMESTEPS):
            # mid.append(cdata.iloc[x: x+TIMESTEPS, :])
            day = cdata.iloc[x: x+TIMESTEPS, :] / cdata.iloc[x+TIMESTEPS-1, -1] - 1
            pdata.append(day.values.flatten())
            plabel.append(cdata.iloc[x+TIMESTEPS, -1] / cdata.iloc[x+TIMESTEPS-1, -1] - 1)
        data = np.asarray(pdata, dtype=np.float32)
        label = np.asarray(plabel, dtype=np.float32)
        np.save('idata_000001', data)
        np.save('ilabel_000001', label)
        print('data saved')

    else:
        try:
            import QUANTAXIS as qa
            candles = qa.QA_fetch_stock_day_adv(stock, start='2014-01-01', end='2018-05-02')
            cdata = candles.data.reset_index(level=1, drop=True).loc[:, pvars]
            # mid = []
            pdata = []
            plabel = []
            for x in range(cdata.shape[0]-TIMESTEPS-3):
                # mid.append(cdata.iloc[x: x+TIMESTEPS, :])
                day = cdata.iloc[x: x+TIMESTEPS, :] / cdata.ix[x+TIMESTEPS-1, 'close'] - 1
                pdata.append(day.values.flatten())
                plabel.append((cdata.ix[x+TIMESTEPS+3, 'close'] / cdata.ix[x+TIMESTEPS, 'open'] - 1) > 0.05)
            data = np.asarray(pdata, dtype=np.float32)
            label = np.asarray(plabel, dtype=np.float32)
            np.save('data_000001', data)
            np.save('label_000001', label)
            print('data saved')
        except Exception as e:
            print(e.message)
    return data, label


def generate_min_data():
    import QUANTAXIS as qa
    stocks = qa.QA_fetch_stock_list_adv()
    train_data = []
    train_label = []
    test_data = []
    # test_label = []

    for stock in stocks.code.tolist():
        print('handling %s' % stock)
        try:
            candles = qa.QA_fetch_stock_min(stock, start='2015-01-01', end='2018-05-08', format='pandas', frequence='5min')
            days = []
            for x in range(candles.shape[0] // INTERVAL):
                day = candles.ix[x*INTERVAL : (x+1)*INTERVAL, ['open', 'close']]
                days.append(np.insert(day.close.values, 0, day.open[0]))
            cdata = pd.DataFrame(days)
            if cdata.shape[0] < 50:
                continue
        except:
            print('data error: {}'.format(stock))
            continue

        pdata = []
        plabel = []
        for x in range(cdata.shape[0]-TIMESTEPS+1):
            # mid.append(cdata.iloc[x: x+TIMESTEPS, :])
            today = cdata.iloc[x: x+TIMESTEPS, :] / cdata.iloc[x+TIMESTEPS-1, -1] - 1
            tmrrw = cdata.iloc[x+TIMESTEPS, -1] / cdata.iloc[x+TIMESTEPS-1, -1] - 1 if x < cdata.shape[0]-TIMESTEPS else 0
            pdata.append(today.values.flatten())
            plabel.append(tmrrw)
        try:
            test = pd.DataFrame(pdata[-TEST_PERIOD:])
            test['code'] = stock
            test['date'] = candles.date.unique()[-TEST_PERIOD:]
            test['label'] = plabel[-TEST_PERIOD:]
            test_data.append(test)
            train_data.extend(pdata[:-TEST_PERIOD])
            train_label.extend(plabel[:-TEST_PERIOD])
        except:
            print('error found: {}'.format(stock))
            continue
        # test_data.extend(pdata[-50:])
        # test_label.extend(plabel[-50:])
    test = pd.concat(test_data)
    # test = test.sort_values('date')
    return np.asarray(train_data, dtype=np.float32), np.asarray(train_label, dtype=np.float32), test


data, label = generate_data('000001')

print(len(data))
# print(data[:2])
# print(label[:2])

train_X = data[:-TEST_PERIOD]
train_y = label[:-TEST_PERIOD]
test_X = data[-TEST_PERIOD:]
test_y = label[-TEST_PERIOD:]

# train_X, train_y, test_df = generate_min_data()
#
# np.save('../data/rnn_train', train_X)
# np.save('../data/rnn_label', train_y)
# test_df.to_hdf('../data/rnn_test.hdf', 'test')
# print('data saved')

n_input = INTERVAL + 1
n_classes = 1

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, HIDDEN_SIZE], stddev=0.1)),  # Hidden layer weights
    'out': tf.Variable(tf.random_normal([HIDDEN_SIZE * 2, n_classes], stddev=0.1))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1), name='biases')
}


w_omega = tf.Variable(tf.random_normal([HIDDEN_SIZE * 2, ATTENTION_SIZE], stddev=0.1))
b_omega = tf.Variable(tf.random_normal([ATTENTION_SIZE], stddev=0.1))
u_omega = tf.Variable(tf.random_normal([ATTENTION_SIZE], stddev=0.1))


def lstm_model(X, y, is_training):

    # 规整成矩阵数据
    X = tf.reshape(X, [-1, TIMESTEPS, n_input])

    ## 规整输入的数据
    X = tf.transpose(X, [1, 0, 2])  # permute n_steps and batch_size
    X = tf.reshape(X, [-1, n_input])  # (n_steps*batch_size, n_input)
    ## 输入层到隐含层，第一次是直接运算
    X = tf.matmul(X, weights['hidden']) + biases['hidden']  # tf.matmul(a,b)   将矩阵a乘以矩阵b，生成a * b

    # 之后使用LSTM

    # 使用多层的LSTM结构。
    # cell = tf.nn.rnn_cell.MultiRNNCell([
    #     tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
    #     for _ in range(NUM_LAYERS)])


    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    # X = tf.split(X, TIMESTEPS, 0)
    X = tf.reshape(X, [-1, TIMESTEPS, HIDDEN_SIZE])
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE),
                                                 tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), X, dtype=tf.float32)
    # output = outputs[:, -1, :]
    with tf.name_scope('Attention_layer'):
        attention_output = attention(outputs, (w_omega, b_omega, u_omega), return_alphas=False)

    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    predictions = tf.matmul(attention_output, weights['out'], name='logits_rnn_out') + biases['out']

    # predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return predictions, None, None

    # 计算损失函数。
    y = tf.reshape(y, [-1, 1])
    # loss = tf.reduce_sum(tf.sqrt(tf.multiply(tf.squared_difference(y, predictions),
    #                                          tf.cast(tf.logical_and(tf.less(y, tf.zeros_like(y) - 0.001),
    #                                                                 tf.greater(predictions, tf.zeros_like(y) + 0.001)), tf.float32) * 4 +
    # #                                 #tf.cast(tf.less(y, predictions), tf.float32) * 1 +
    #                              tf.ones_like(y))))
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # pred = tf.argmax(predictions, 1)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adam", learning_rate=0.001)
    return predictions, loss, train_op


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



def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(len(test_X)):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)

    # 对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions * 10, label='predictions')
    plt.plot(labels, label='real_close')
    plt.legend()
    plt.show()


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
    # print("Evaluate model before training.")
    # run_eval(sess, test_X, test_y)

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
    print("Evaluate model after training.")
    run_eval(sess, test_X, test_y)
