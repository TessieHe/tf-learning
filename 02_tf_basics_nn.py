# coding: utf-8！

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

print('tf.__version__:',tf.__version__)


def add_layer(inputs,input_size,output_size,n_layer,active_func=None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_uniform([input_size,output_size],0,1),name='w')
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([1,output_size])+0.1,name='b')

        with tf.name_scope('logits'):
            w_plus =tf.add( tf.matmul(inputs,w) , b)
        if active_func is None:
            outputs = w_plus
        else:
            outputs = active_func(w_plus)
    tf.summary.histogram(layer_name + '/weights', w)
    tf.summary.histogram(layer_name + '/biases', b)
    tf.summary.histogram(layer_name + '/ouputs', outputs)

    return outputs

# create data

x_data = np.linspace(-1,1,300).astype(np.float32)[:,np.newaxis]
print('x_data.shape',x_data.shape)
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

# create tf structures
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
layer1 =  add_layer(xs,x_data.shape[1],10,1,active_func=tf.nn.relu)
layer3 =  add_layer(layer1,10,1,2,active_func=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(layer3-ys))
    #tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# train
sess = tf.Session()

tf.summary.scalar('loss',loss)
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/',sess.graph) #tensorborad

sess.run(init)

for step in range(1000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if step %50 ==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass


        y_hat,l = sess.run((layer3,loss),feed_dict={xs:x_data,ys:y_data})
        print('step:{}  loss:{}'.format(step,l))
        lines = ax.plot(x_data,y_hat,'r-',lw=5)
        plt.pause(0.1)
        res = sess.run(merge,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(res,step)

sess.close()


plt.pause(0)