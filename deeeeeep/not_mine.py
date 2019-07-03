import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/",one_hot=True)

learning_rate = 0.001
training_iters = 40000
batch_size = 100
display_step = 10
n_input = 784
n_classes = 10


x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
weights = {
    'wf':tf.Variable(tf.truncated_normal([784, 1024], stddev=0.1)),
    'out':tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
}
biases = {
    'bf':tf.Variable(tf.constant(0.1, shape=[1024])),
    'out':tf.Variable(tf.constant(0.1, shape=[10]))
}


def bp_net(_X,_weights,_biases):
    _X = tf.reshape(_X,[-1,28,28,1])
    dense1 = tf.reshape(_X,[-1,_weights['wf'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wf']),_biases['bf']))
    # out = tf.nn.softmax(tf.add(tf.matmul(dense1,_weights['out']),_biases['out']))
    out = tf.add(tf.matmul(dense1,_weights['out']),_biases['out'])
    print(out)
    return out


pred = bp_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
# cost = -tf.reduce_max(y * tf.log(pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size<training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict = {x:batch_xs,y:batch_ys})
        if step %display_step==0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256]}))
