# coding: utf-8！

import tensorflow as tf
import numpy as np

print('tensorflow version : ',tf.__version__  )

# create data
x_data=np.random.rand(100).astype(np.float32)
y_data = x_data*0.3+0.4

### create tensorflow structure start ###

w = tf.Variable(tf.random_uniform([1],0,1),name='w')
b = tf.Variable(tf.random_uniform([1],0,1),name = 'b')
y_hat = x_data * w + b
loss = tf.reduce_mean(tf.square(y_data-y_hat),name='loss')

opimizer = tf.train.GradientDescentOptimizer(0.5)
train  = opimizer.minimize(loss)

init = tf.initialize_all_variables()

### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)
print('w: ',w)

for step in range(50):
    sess.run(train)
    if step %5 == 0:
        loss_,w_,b_=sess.run([loss,w,b])
        print('step {}    loss:{} , w:{}, b:{}'.format(step,loss_,w_,b_))

sess.close()