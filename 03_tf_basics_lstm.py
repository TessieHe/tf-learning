# coding: utf-8！
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# this is data
print('loading data')
mnist = input_data.read_data_sets('/Users/tessiehe/Documents/charging/tf_basics/data/mnist/',one_hot=True)

# hyperparameter
lr = 0.001
training_iters = 1000
batch_size = 128

n_inputs = 28 # MNIST数据实28*28,每行实一个step
n_steps = 28
n_hidden_units= 128 # neurons in hidden layer
n_classes = 10 #10分类

# tf Graph input
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x')
y = tf.placeholder(tf.float32,[None,n_classes],name='y')

# define weights
weights={
    # (28,128)
    'w_x': tf.Variable(tf.random_normal([n_inputs,n_hidden_units]),name='w_x'),
    # (128,10)
    'w_y': tf.Variable(tf.random_normal([n_hidden_units,n_classes]),name='w_y'),
    # (128,128)
    #'w_a': tf.Variable(tf.random_normal([n_hidden_units,n_hidden_units]),name='w_a') 在cell中封装好
}

biases = {
    # (1,128)
    'b_x' : tf.Variable(tf.constant(0.1,shape=[1,n_hidden_units])),

    # (1,10)
    'b_y' : tf.Variable(tf.constant(0.1,shape=[1,n_classes]))
}

def LSTM(X,weights,biases):
    # hidden layer
    ##################################################
    # X(128 batch size, 28 step, 28 inputs)
    # ---> (128*28,18)??
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['w_x'])+biases['b_x'] #(128*28,128)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    # cell
    ##################################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    # output
    ##################################################
    res = tf.matmul(states[1],weights['w_y'])+biases['b_y']  # softmax 之前的logit
    return res


pred = LSTM(x,weights,biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()


def makedir(dir):
    if not  os.path.exists(dir):
        os.system(r'mkdir -p {}'.format(dir))


with tf.Session() as sess:
    makedir('./logs/lstm/')
    writer = tf.summary.FileWriter('./logs/lstm/', sess.graph)
    print('training')
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        step+=1
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        if step % 5 == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
            ll = sess.run(loss,feed_dict={x:batch_xs,y:batch_ys})
            print('step:{}  accuracy:{} loss:{}'.format(step,acc,ll))