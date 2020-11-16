import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import tensorlayer as tl


#记录程序运行时间
import time
start_time = time.time()

tt = pd.read_csv("../data/train_15_17.csv", nrows=10)
nvars = tt.shape[1] - 3
names = ['date'] + ["v{}".format(x) for x in range(nvars)] + ['label', 'code']
# names = ['date'] + ["v{}".format(x) for x in range(nvars)] + ['p10', 'p0', 'n0', 'n10', 'code']

#读入数据
train = pd.read_csv("../data/train_15_17.csv", skiprows=1, names=names)

train_xy, val = train_test_split(train, test_size=0.2, random_state=999)
# random_state is of big influence for val-auc
train_y = np.asarray(train_xy.label)
train_X = np.asarray(train_xy.drop(['label', 'date', 'code'], axis=1))
val_y = np.asarray(val.label)
val_X = np.asarray(val.drop(['label', 'date', 'code'], axis=1))

tests = pd.read_csv("../data/test_15_17.csv", skiprows=1, names=names)
test_X = np.asarray(tests.drop(['label', 'date', 'code'], axis=1))
test_y = np.asarray(tests.label)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 150], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

network = tl.layers.InputLayer(inputs=x, name='input_layer')
network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
network = tl.layers.DenseLayer(network, n_units=128, act=tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
network = tl.layers.DenseLayer(network, n_units=4, act=tf.identity, name='output_layer')

y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='cost')
correct = tf.equal(tf.arg_max(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
y_op = tf.arg_max(tf.nn.softmax(y), 1)

train_param = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, use_locking=False).minimize(cost, var_list=train_param)

acc_summ = tf.summary.scalar('acc', acc)
cost_summ = tf.summary.scalar('cost', cost)
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs')
writer.add_graph(sess.graph)

tl.layers.initialize_global_variables(sess)

network.print_layers()
network.print_params()

tl.utils.fit(sess, network, train_op, cost, train_X, train_y, x, y_,
             acc=acc, batch_size=128, n_epoch=2000, print_freq=10,
             X_val=val_X, y_val=val_y, eval_train=False, tensorboard=True)

tl.utils.test(sess, network, acc, test_X, test_y, x, y_, batch_size=None, cost=cost)

tl.files.save_npz(network.all_params, name='tl_dense.npz')
sess.close()

# tests = tests.loc[:, ['date', 'code', 'label']]
# tests['pred'] = pd.Series(preds)
# tests.to_csv('../data/dense_pred.csv')

#输出运行时长
cost_time = time.time()-start_time
print("dense success!", '\n', "cost time:", cost_time, "(s)")
