# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
import xlrd
import xlwt
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import signal
import tensorflow as tf
import os # 决策树可视化Graphviz的环境变量没用，因此直接在工程导入
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

fs = 1000
lowcut = 40
highcut = 300
order = 1
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = signal.butter(order, [low, high], btype='band')
lv = 4 # 小波滤波层数
m = 0 # 起始层数
n = 2 # 结束层数
wavefunc = 'db4'#小波变换方式


def divide_data(input_path,num,data_type):
    for order in range(num):
        origin_data = np.loadtxt(input_path+'%d.txt'%order)
        plt.plot(signal.lfilter(b, a,origin_data))
        plt.show()
        str_in = input('请输入数据位置：')
        position = [int(n) for n in str_in.split(' ')]
        if position[1] == 0:
            print ('Noise')
            continue
        elif position[1] == 1:
            print ('Hard to divide')
            continue
        if len(position) == 2:
            former_part = origin_data[position[0]:position[1]]
            else_part = np.append(origin_data[0:position[0]], origin_data[position[1]:len(origin_data)])
        elif len(position) == 4:
            former_part = np.append(origin_data[position[0]:position[1]],origin_data[position[2]:position[3]])
            temp = np.append(origin_data[0:position[0]],origin_data[position[1]:position[2]])
            else_part = np.append(temp,origin_data[position[3]:len(origin_data)])
        np.savetxt(input_path+'%d_%d.txt'%(order,data_type),former_part)
        np.savetxt(input_path+'%d_0.txt'%order,else_part)


def generate_eigenvalue(input_path,output_path,file_num,class_name,slide_length,interval,property_num,use_filter):
    eigen_list = [[0 for x in range(property_num+1)]]
    for files in range(file_num):
        if use_filter == 1:
            data = signal.lfilter(b, a, np.loadtxt(input_path+'%d_%d.txt' % (files,class_name)))
        else:
            data = np.loadtxt(input_path+'%d_%d.txt' % (files,class_name))
        for i in range((len(data)-slide_length+interval)/interval):
            if (i*interval+slide_length)>len(data):
                slide = data[(len(data)-slide_length):len(data)]
                eigenvalue = cal_eigenvalue(slide,class_name)
                eigen_list.append(eigenvalue)
                break
            slide = data[i*interval:(i*interval+slide_length)]
            eigenvalue = cal_eigenvalue(slide,class_name)
            eigen_list.append(eigenvalue)
        print ('one done')
    file_excel = xlwt.Workbook()
    file_excel_sheet1 = file_excel.add_sheet(u'sheet1', cell_overwrite_ok=True)
    for rows in range(len(eigen_list)-1):
        for cols in range(property_num+1):
            file_excel_sheet1.write(rows, cols, float(eigen_list[rows+1][cols]))
    file_excel.save(output_path)
    print ('xls done')


def cal_eigenvalue(data,class_name):
    fr = abs(np.fft.fft(data))
    fr = fr[1:1000]
    energy = 0.
    Kurtosis = 0.
    Skewness = 0.
    Kurtosis_fr = 0.
    Skewness_fr = 0.
    fr_50 = fr[0:200]
    fr_150 = fr[200:600]
    len_data = len(data)
    len_fr = len(fr)
    const = float(len_data)/((len_data-1)*(len_data-2))
    const_fr = float(len_fr) / ((len_fr - 1) * (len_fr - 2))
    for i in range(len(data)):
        energy += data[i]*data[i]
    mean = np.mean(data)
    mean_fr = np.mean(fr)
    std = np.std(data)
    for j in range(len_data):
        Kurtosis += np.power((data[j]-mean)/std,4)
        Skewness += np.power((data[j]-mean),3)
    std_fr = np.std(fr)
    for j in range(len_fr):
        Kurtosis_fr += np.power((fr[j]-mean_fr)/std_fr,4)
        Skewness_fr += np.power((fr[j]-mean_fr),3)
    Skewness = const*Skewness/np.power(std,3)
    Skewness_fr = const_fr * Skewness_fr / np.power(std_fr, 3)
    mean_50 = np.mean(fr_50)
    mean_150 = np.mean(fr_150)
    return [energy,Kurtosis/len_data,Skewness,Kurtosis_fr/len_fr,Skewness_fr,mean_50,mean_150,class_name]
# def cal_eigenvalue(data,threshold,class_name):
#     count_zcr = 0.
#     count_ratio = 0.
#     Kurtosis = 0.
#     Skewness = 0.
#     len_data = len(data)
#     const = float(len_data)/((len_data-1)*(len_data-2))
#     max = np.max(data)
#     mean = np.mean(data)
#     std = np.std(data)
#     for i in range(len_data-1):
#         if data[i] * data[i + 1] < 0:
#             count_zcr += 1
#     for j in range(len_data):
#         Kurtosis += np.power((data[j]-mean)/std,4)
#         Skewness += np.power((data[j]-mean),3)
#         if abs(data[j])<threshold:
#             # data[j] = 0
#             count_ratio += 1
#     Skewness = const*Skewness/np.power(std,3)
#     mean = np.mean(abs(data))
#     return [max,mean,count_zcr/len(data),1-count_ratio/len(data),Kurtosis/len(data),Skewness,class_name]


def xls_tolist_python(xls,property_num,data_len):     #定义函数将读取的表格数据转换为list数据，输入表格，列数，行数
    temp = [[0 for i in range(property_num)] for j in range(data_len)]       #建立空list，存放接下来的表格数据
    for rows in range(data_len):     #遍历表格数据行数
        for cols in range(property_num):  #便利表格列数
            temp[rows][cols] = xls.cell(rows , cols ).value   #读取数据，赋值
    result = np.array(temp)     #最终结果为python的list，将其转化为np的list，方便处理
    return result       #返回list数据


# def filter_by_wavelet(plt_data): #定义函数执行小波滤波，输入为数据帧
#     coeff = pywt.wavedec(plt_data, wavefunc, mode='sym', level=lv, axis=0) #将滤波后的小波数据放在coeff中
#     sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # 非线性函数，用于收缩数据值
#     for i in range(m, n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
#         cD = coeff[i]   # 分解后的小波系数
#         for j in range(len(cD)): #遍历cd
#             Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
#             if cD[j] >= Tr:
#                 coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
#             else:
#                 coeff[i][j] = 0  # 低于阈值置零
#     denoised_index = pywt.waverec(coeff, wavefunc) #反变换回时域
#     return denoised_index


def change_label_cols(data):
    label1 = data[:,-1]
    label2 = np.copy(label1)
    label3 = np.copy(label1)
    test = data.shape
    for rows in range(data.shape[0]):
        if data[rows][7] == 0:
            label1[rows] = 1
            label2[rows] = 0
            label3[rows] = 0
        elif data[rows][-1] == 1:
            label1[rows] = 0
            label2[rows] = 1
            label3[rows] = 0
        else:
            label1[rows] = 0
            label2[rows] = 0
            label3[rows] = 1
    data = np.delete(data,-1,axis = 1)
    data = np.c_[data,label1,label2,label3]
    # np.insert(data,-1, values=label1, axis=1)
    # np.insert(data, -1, values=label2, axis=1)
    # np.insert(data, -1, values=label3, axis=1)
    return data


# path = ''
# divide_data(path,1,2)
# generate_eigenvalue(path,r'eigen/car_4_2.xls',file_num=1,class_name=2,slide_length=2000,
#                     interval=200,property_num=7,use_filter=0)
workbook1 = xlrd.open_workbook(r'唐森/car.xls')
sheet_train1 = workbook1.sheet_by_name('sheet1')
x1 = xls_tolist_python(sheet_train1,8,172)

workbook2 = xlrd.open_workbook(r'唐森/man.xls')
sheet_train1 = workbook2.sheet_by_name('sheet1')
x2 = xls_tolist_python(sheet_train1,8,242)

workbook3 = xlrd.open_workbook(r'唐森/en2.xls')
sheet_train1 = workbook3.sheet_by_name('sheet1')
x3 = xls_tolist_python(sheet_train1,8,236)

workbook4 = xlrd.open_workbook(r'唐森/en_.xls')
sheet_train1 = workbook4.sheet_by_name('sheet1')
x4 = xls_tolist_python(sheet_train1,8,100)
# workbook2 = xlrd.open_workbook(r'eigen/en.xls')
# sheet_train2 = workbook2.sheet_by_name('sheet1')
# x2 = xls_tolist_python(sheet_train2,8,200)
#
# workbook3 = xlrd.open_workbook(r'eigen/man.xls')
# sheet_train3 = workbook3.sheet_by_name('sheet1')
# x3 = xls_tolist_python(sheet_train3,8,100)
#
# workbook4 = xlrd.open_workbook(r'eigen/en_wind.xls')
# sheet_train4 = workbook4.sheet_by_name('sheet1')
# x4 = xls_tolist_python(sheet_train4,8,200)
#
# workbook5 = xlrd.open_workbook(r'eigen/man_10m.xls')
# sheet_train5 = workbook5.sheet_by_name('sheet1')
# x5 = xls_tolist_python(sheet_train5,8,100)
#
# workbook6 = xlrd.open_workbook(r'eigen/man_30m.xls')
# sheet_train6 = workbook6.sheet_by_name('sheet1')
# x6 = xls_tolist_python(sheet_train6,8,100)
#
# workbook = xlrd.open_workbook(r'eigen/man3.xls')
# sheet_train = workbook.sheet_by_name('sheet1')
# y1 = xls_tolist_python(sheet_train,8,50)
#
# workbook = xlrd.open_workbook(r'eigen/car3.xls')
# sheet_train = workbook.sheet_by_name('sheet1')
# y2 = xls_tolist_python(sheet_train,8,30)
#
# workbook = xlrd.open_workbook(r'eigen/en3.xls')
# sheet_train = workbook.sheet_by_name('sheet1')
# y3 = xls_tolist_python(sheet_train,8,100)
#
# workbook = xlrd.open_workbook(r'eigen/en4.xls')
# sheet_train = workbook.sheet_by_name('sheet1')
# y4 = xls_tolist_python(sheet_train,8,20)
#
# workbook = xlrd.open_workbook(r'eigen/car4.xls')
# sheet_train = workbook.sheet_by_name('sheet1')
# y5 = xls_tolist_python(sheet_train,8,20)
#
# workbook = xlrd.open_workbook(r'eigen/man4.xls')
# sheet_train = workbook.sheet_by_name('sheet1')
# y6 = xls_tolist_python(sheet_train,8,40)
#
# workbook1 = xlrd.open_workbook(r'eigen/car_4_2.xls')
# sheet_train1 = workbook1.sheet_by_name('sheet1')
# y7 = xls_tolist_python(sheet_train1,8,100)
# # 早期人车
# y = np.concatenate((y4,y5,y6,y7,y8,y9),axis=0)
# data_set = np.concatenate((x1,x2,x3,x4,x5,x6),axis=0)
# 后院
# y = np.concatenate((y8,y7),axis=0)
# data_set = np.concatenate((x1,x2,x3,x4,x5,x6),axis=0)
# 近期数据
# y = np.concatenate((y1,y5,y3,x4),axis=0)
# data_set = np.concatenate((x1,x2,x3,x5,x6),axis=0)

data_set = np.concatenate((x1,x2,x4),axis=0)
X,Y = train_test_split(data_set,test_size=0.2)
X_train = X[:,0:-1]
X_label = X[:,-1]
# clf = SVC(C = 0.8,gamma='auto',kernel='linear')
clf = tree.DecisionTreeClassifier()

clf.fit(X_train, X_label)  # 训练模型
types = clf.predict(Y[:,0:-1])
pos = 0.
for num in range(len(Y)):
    if Y[num][-1] == types[num]:
        pos += 1
print('训练集精确度：')
print(pos / len(Y))
# print(types)
Matrix = confusion_matrix(Y[:, -1], types)
print(Matrix)




#
#
# types = clf.predict(y[:,0:-1])
# pos = 0.
# for num in range(len(y)):
#     if y[num][-1]==types[num]:
#        pos+=1
# print ('测试集精确度：')
# print (pos/len(y))
# Matrix = confusion_matrix(y[:,-1],types)
# print (Matrix)
# dot_data = tree.export_graphviz(clf, out_file=None)
# #可视化部分，可直接屏蔽
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("tree_new.png")


# learning_rate = 0.01
# training_iters = 300000
# batch_size = 100
# display_step = 100
# n_input = data_num
# n_classes = 3
# # 训练集
# origin_data = change_label_cols(data_set)
# np.random.shuffle(origin_data)
# features = origin_data[:-100,0:-3]
# labels = origin_data[:-100,-3:]
# assert features.shape[0] == labels.shape[0]
# features_placeholder = tf.placeholder(tf.float32, [None,features.shape[1]])
# labels_placeholder = tf.placeholder(tf.float32, [None,3])
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# # iterator = dataset.make_initializable_iterator()
# # x = tf.placeholder(tf.float32,[None,n_input])
# # y = tf.placeholder(tf.float32,[None,n_classes])
# weights = {
#     'wf':tf.Variable(tf.truncated_normal([7, 200], stddev=0.1)),
#     'out':tf.Variable(tf.truncated_normal([200,3], stddev=0.1))
# }
# biases = {
#     'bf':tf.Variable(tf.constant(0.1, shape=[200])),
#     'out':tf.Variable(tf.constant(0.1, shape=[3]))
# }
#
#
# def bp_net(_X,_weights,_biases):
#     _X = tf.reshape(_X,[-1,1,7,1])
#     temp = _weights['wf'].get_shape().as_list()[0]
#     dense1 = tf.reshape(_X,[-1,_weights['wf'].get_shape().as_list()[0]])
#     dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wf']),_biases['bf']))
#     # out = tf.nn.softmax(tf.add(tf.matmul(dense1,_weights['out']),_biases['out']))
#     out = tf.add(tf.matmul(dense1,_weights['out']),_biases['out'])
#     print(out)
#     return out
#
#
# pred = bp_net(features_placeholder, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred))
# # cost = -tf.reduce_max(y * tf.log(pred))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#
# correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(labels_placeholder,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
# init = tf.initialize_all_variables()
# loss_hist = tf.summary.scalar('loss', cost)
# acc_hist = tf.summary.scalar('accuracy', accuracy)
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     merged = tf.summary.merge_all()
#     # 同样写入到logs中
#     writer = tf.summary.FileWriter('./logs/summary', sess.graph)
#     while step * batch_size<training_iters:
#         # batch_xs,batch_ys = dataset.train.next_batch(batch_size)
#         summary, _, l = sess.run(
#             [merged, optimizer, cost],
#             feed_dict={features_placeholder: features, labels_placeholder: labels})
#         # sess.run(optimizer,feed_dict={features_placeholder: features, labels_placeholder: labels})
#         if step % display_step==0:
#             acc = sess.run(accuracy,feed_dict={features_placeholder: features, labels_placeholder: labels})
#             loss = sess.run(cost, feed_dict={features_placeholder: features, labels_placeholder: labels})
#             print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss)
#                   + ", Training Accuracy= " + "{:.5f}".format(acc))
#             writer.add_summary(summary, step)
#         step += 1
#     print("Optimization Finished!")
#     print("Testing Accuracy:", sess.run(accuracy, feed_dict={features_placeholder: origin_data[-100:-1,0:-3],
#                                                              labels_placeholder: origin_data[-100:-1,-3:]}))
