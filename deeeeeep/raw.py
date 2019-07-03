import numpy as np
import tensorflow as tf
import sklearn


raw_data = {'path':['sensor/deep/car/','sensor/deep/en/','sensor/deep/man/'],
            'fnum':[7,5,14],
            'label':[2,0,1]}


def change_label_cols(input_data):
    label1 = input_data[:,-1]
    label2 = np.copy(label1)
    label3 = np.copy(label1)
    for rows in range(input_data.shape[0]):
        if input_data[rows][7] == 0:
            label1[rows] = 1
            label2[rows] = 0
            label3[rows] = 0
        elif input_data[rows][-1] == 1:
            label1[rows] = 0
            label2[rows] = 1
            label3[rows] = 0
        else:
            label1[rows] = 0
            label2[rows] = 0
            label3[rows] = 1
        input_data = np.delete(input_data,-1,axis = 1)
        input_data = np.c_[input_data,label1,label2,label3]
    return input_data


def load_data(dic,slide_len):
    dataset = [[]]
    label_ = [0,0,0]
    count_out = 0
    for path in dic['path']:
        count_in = 0
        for name in dic['fnum']:
            origin_data = np.loadtxt(path+'disposed/'+str(name)+'.txt')
            for rows in range(int(len(origin_data)/slide_len)):
                if rows*slide_len>=len(origin_data):
                    dataset.append(origin_data[-slide_len:])
                else:
                    dataset.append(origin_data[rows*slide_len:(rows+1)*slide_len])
                count_in += 1
        label_[count_out] = count_in
        count_out += 1
    dataset = dataset[1:]
    label1 = [2 for i in range(label_[0])]
    label2 = [0 for i in range(label_[1])]
    label3 = [1 for i in range(label_[2])]
    dataset = np.c_[dataset,np.r_[label1,label2,label3]]
    return dataset


def divide_data(dic):
    for path in dic['path']:
        pos = 0
        for name in range(dic['fnum'][pos]):
            count = 0
            origin_data = np.loadtxt(path+'%d' % name+'.txt')
            str_in = input('请输入数据位置：')
            position = [int(n) for n in str_in.split(' ')]
            if position[1] == 'pass':
                print('Ignore this file')
                count += 1
            elif len(position) == 2:
                useful_data = origin_data[position[0]:position[1]]
                np.savetxt(fname=path + 'disposed/' + str(name-count) + '.txt', X=useful_data, fmt='%.4f')
            elif len(position) == 4:
                useful_data = origin_data[position[0]:position[1]].append(
                                             origin_data[position[2]:position[3]])
                np.savetxt(fname=path + 'disposed/' + str(name-count) + '.txt', X=useful_data, fmt='%.4f')
        pos += 1


def bp_net(_X,_weights,_biases):
    _X = tf.reshape(_X,[-1,1,7,1])
    dense1 = tf.reshape(_X,[-1,_weights['wf'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wf']),_biases['bf']))
    # out = tf.nn.softmax(tf.add(tf.matmul(dense1,_weights['out']),_biases['out']))
    out = tf.add(tf.matmul(dense1,_weights['out']),_biases['out'])
    print(out)
    return out


def cnn_net(_X,_weights,_biases):

    _X = tf.reshape(_X, [-1, 1, 2000, 1])
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    conv1 = max_pooling(conv1, k=10)# 这里可能要改 注意一下

    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pooling(conv2, k=10)# 这里可能要改 注意一下

    dense = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, _weights['wd1']), _biases['bd1']))

    out = tf.add(tf.matmul(dense, _weights['out']), _biases['out'])
    print(out)
    return out


def conv2d(image,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image,w,strides=[1,1,1,1],padding='SAME'),b))


def max_pooling(image,k):
    return tf.nn.max_pool(image, ksize=[1,1,k,1], strides=[1,1,k,1], padding='SAME')


# 数据处理及分割
# divide_data(raw_data)
data_set = load_data(raw_data,2000)
# X,Y = sklearn.model_selection.train_test_split(data_set,test_size=0.2)
# X_train = X[:,0:-1]
# X_label = X[:,-1]
# data_num = X_train.shape[0]

# 超参数及其他
learning_rate = 0.01
training_iters = 200000
batch_size = 100
display_step = 100
# n_input = data_num
n_classes = 3
conv_len = 200
# 网络搭建部分

# bp网络参数
# weights = {
#     'wf':tf.Variable(tf.truncated_normal([7, 1024], stddev=0.1)),
#     'out':tf.Variable(tf.truncated_normal([1024,3], stddev=0.1))
# }
# biases = {
#     'bf':tf.Variable(tf.constant(0.1, shape=[1024])),
#     'out':tf.Variable(tf.constant(0.1, shape=[3]))
# }
# cnn参数
weights = {
    'wc1':tf.Variable(tf.truncated_normal([1,conv_len,1,32],stddev=0.1)),
    'wc2':tf.Variable(tf.truncated_normal([1,conv_len,32,64])),
    'wd1':tf.Variable(tf.truncated_normal([1*20*64,1024])),
    'out':tf.Variable(tf.truncated_normal([1024,n_classes]))
}
biases = {
    'bc1':tf.Variable(tf.constant(0.1,shape=[32])),
    'bc2':tf.Variable(tf.constant(0.1,shape=[64])),
    'bd1':tf.Variable(tf.constant(0.1,shape=[1024])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes]))
}

# 训练集
data = change_label_cols(data_set)
np.random.shuffle(data)
features = data[:-100,0:-3]
labels = data[:-100,-3:]
assert features.shape[0] == labels.shape[0]
features_placeholder = tf.placeholder(tf.float32, [None,features.shape[1]])
labels_placeholder = tf.placeholder(tf.float32, [None,3])
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# iterator = dataset.make_initializable_iterator()
# x = tf.placeholder(tf.float32,[None,n_input])
# y = tf.placeholder(tf.float32,[None,n_classes])

# pred = bp_net(features_placeholder, weights, biases)
pred = cnn_net(features_placeholder, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred))
# cost = -tf.reduce_max(y * tf.log(pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.initialize_all_variables()

#  运行
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size<training_iters:
        # batch_xs,batch_ys = dataset.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={features_placeholder: features, labels_placeholder: labels})
        if step % display_step==0:
            acc = sess.run(accuracy,feed_dict={features_placeholder: features, labels_placeholder: labels})
            loss = sess.run(cost, feed_dict={features_placeholder: features, labels_placeholder: labels})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss)
                  + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={features_placeholder: data[-100:-1,0:-3],
                                                             labels_placeholder: data[-100:-1,-3:]}))
