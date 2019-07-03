import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/",one_hot=True)
#Parameters
learning_rate = 0.001
training_iters = 60000
batch_size = 200
display_step = 10
n_input = 784
n_classes = 10
dropout = 0.8

#  输入
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# 参数定义及赋值
def conv2d(image,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pooling(image,k):
    return tf.nn.max_pool(image, ksize=[1,k,1,1], strides=[1,k,k,1], padding='SAME')

weights = {
    'wc1':tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1)),
    'wc2':tf.Variable(tf.truncated_normal([5,5,32,64])),
    'wd1':tf.Variable(tf.truncated_normal([7*7*64,1024])),
    'out':tf.Variable(tf.truncated_normal([1024,n_classes]))
}
biases = {
    'bc1':tf.Variable(tf.constant(0.1,shape=[32])),
    'bc2':tf.Variable(tf.constant(0.1,shape=[64])),
    'bd1':tf.Variable(tf.constant(0.1,shape=[1024])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes]))
}
# weights = {
#     'wf':tf.Variable(tf.truncated_normal([784, 1024], stddev=0.1)),
#     'out':tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
# }
# biases = {
#     'bf':tf.Variable(tf.constant(0.1, shape=[1024])),
#     'out':tf.Variable(tf.constant(0.1, shape=[10]))
# }


def conv_net(_X,_weights,_biases):
    #第一卷积层
    _X = tf.reshape(_X,[-1,28,28,1])
    conv1 = conv2d(_X,_weights['wc1'],_biases['bc1'])
    conv1 = max_pooling(conv1, k = 2)

    #第二卷积层
    conv2 = conv2d(conv1,_weights['wc2'],_biases['bc2'])
    conv2 = max_pooling(conv2, k=2)

    #全连接
    dense = tf.reshape(conv2,[-1,_weights['wd1'].get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense,_weights['wd1']),_biases['bd1']))
    # 输出
    out = tf.add(tf.matmul(dense,_weights['out']),_biases['out'])
    print(out)
    return out
# 预测
pred = conv_net(x, weights, biases)
# cost = -tf.reduce_mean(y * tf.log(pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.initialize_all_variables()
# tf图开始运行
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size<training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict = {x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step %display_step==0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
