import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import tensorlayer as tl


tt = pd.read_csv("../data/train_16_17.csv", nrows=10)
nvars = tt.shape[1] - 3
names = ['date'] + ["v{}".format(x) for x in range(nvars)] + ['label', 'code']

#读入数据
train = pd.read_csv("../data/train_16_17.csv", skiprows=1, names=names)

train_xy, val = train_test_split(train, test_size=0.2, random_state=999)
# random_state is of big influence for val-auc
y_train = np.asarray(train_xy.label)
X_train = np.asarray(train_xy.drop(['label', 'date', 'code'], axis=1))
y_val = np.asarray(val.label)
X_val = np.asarray(val.drop(['label', 'date', 'code'], axis=1))

tests = pd.read_csv("../data/test_16_17.csv", skiprows=1, names=names)
X_test = np.asarray(tests.drop(['label', 'date', 'code'], axis=1))
y_test = np.asarray(tests.label)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 150], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

act = tf.nn.relu
act_recon = tf.nn.softplus

# Define net
print("\nBuild net")
net = tl.layers.InputLayer(x, name='input')
# denoise layer for AE
net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')
# 1st layer
net = tl.layers.DropoutLayer(net, keep=0.8, name='drop1')
net = tl.layers.DenseLayer(net, n_units=256, act=act, name='encoder1')
x_recon1 = net.outputs
recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=150, act=act_recon, name='recon_layer1')
# 2nd layer
net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
net = tl.layers.DenseLayer(net, n_units=256, act=act, name='encoder2')
recon_layer2 = tl.layers.ReconLayer(net, x_recon=x_recon1, n_units=256, act=act_recon, name='recon_layer2')
# 3rd layer
net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')
net = tl.layers.DenseLayer(net, 4, act=tf.identity, name='output')

# Define fine-tune process
y = net.outputs
cost = tl.cost.cross_entropy(y, y_, name='cost')

n_epoch = 1000
batch_size = 128
learning_rate = 0.0001
print_freq = 20

train_params = net.all_params

# train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

# Initialize all variables including weights, biases and the variables in train_op
tl.layers.initialize_global_variables(sess)

# Pre-train
print("\nAll net Params before pre-train")
net.print_params()
print("\nPre-train Layer 1")
recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1',
                      n_epoch=100, batch_size=128, print_freq=10, save=False)
print("\nPre-train Layer 2")
recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1',
                      n_epoch=100, batch_size=128, print_freq=10, save=False)
print("\nAll net Params after pre-train")
net.print_params()

# Fine-tune
print("\nFine-tune net")
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update(net.all_drop)  # enable noise layers
        feed_dict[tl.layers.LayersConfig.set_keep['denoising1']] = 1  # disable denoising layer
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
            feed_dict = {x: X_val_a, y_: y_val_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))
        # try:
        #     # visualize the 1st hidden layer during fine-tune
        #     tl.vis.draw_weights(net.all_params[0].eval(), second=10, saveable=True, shape=[28, 28], name='w1_' + str(epoch + 1), fig_idx=2012)
        # except:  # pylint: disable=bare-except
        #     print("You should change vis.draw_weights(), if you want to save the feature images for different dataset")

print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
    feed_dict = {x: X_test_a, y_: y_test_a}
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err
    test_acc += ac
    n_batch += 1
print("   test loss: %f" % (test_loss / n_batch))
print("   test acc: %f" % (test_acc / n_batch))
# print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))

# Add ops to save and restore all the variables.
# ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
saver = tf.train.Saver()
# you may want to save the model
save_path = saver.save(sess, "../model/encoder.ckpt")
print("Model saved in file: %s" % save_path)
sess.close()

