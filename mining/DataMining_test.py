# -*- coding: utf-8 -*-     支持文件中出现中文字符
import numpy as np
import matplotlib .pyplot as plt
from prettytable import PrettyTable

origin_data_x = np.loadtxt('new_data/20151026_113.txt')  # 加载数据113
origin_data_y = np.loadtxt('new_data/20151026_114.txt')  # 加载114
X = origin_data_x[:, 0]  # get the first col of data
Y = origin_data_y[:, 0]  # get the first col of data
# def cal_euclidean_distance(x, y):  # calculate the euclidean distance of data
#     return np.sqrt(np.sum(np.square(x - y)))
#
#
# def cal_minkowski_distance(x, y, n):  # calculate the minkowski distance of data
#     return np.power(np.sum(np.power(np.abs(x-y), n)), 1.0/n)
#
#
# def cal_cosine_similarity(x, y):  # calculate the cos similarity of data
#     return np.dot(x, y)*1.0/(np.linalg.norm(x)*np.linalg.norm(y))
#
#
# print 'euclidean distance:', cal_euclidean_distance(X, Y)
# print 'minkowski distance:', cal_minkowski_distance(X, Y, 3)
# print 'cos similarity:', cal_cosine_similarity(X, Y)

plt.plot(range(len(X)),X)
plt.show()
def cal_mean(data):
    return np.mean(data)   # 调用numpy方法mean计算均值


def cal_range(data):
    return np.max(data)-np.min(data)   # 调用求最大最小值方法计算极差


def cal_var(data):
    return np.var(data)    # 调用numpy方法计算方差


def cal_aad(data):
    temp = np.mean(data)  # 取中间变量存放均值
    aad = 0               # 求和变量
    for i in range(len(data)):  # 遍历数据，按照公式求取aad
        aad = aad+abs(data[i]-temp)
    return 1.0*aad/len(data)


def cal_frequency(data):  # 定义函数求大于0.1数据的频率
    temp = 0  # 定义中间变量用于计数
    for i in range(len(data)):  # 遍历数据，找到所有大于0.1的值
        if data[i] >= 0.1:
            temp = temp + 1
    return temp*1.0/len(data)


def cal_cut_mean(data):  # 定义函数计算小于0.1的数值的均值
    summary = 0  # 求和变量，用于和计数变量取商计算均值
    temp = 0  # 计数变量，用于存储小于0.1数值的个数
    for i in range(len(data)):
        if data[i]<=0.1:
            summary = summary+data[i]
            temp = temp+1
    return summary*1.0/temp


x = PrettyTable(["Tid", "均值", "极差", "方差","小于0.1的数值均值","大于0.1的数值频率","绝对平均偏差"]) # 定义一个表格
x.padding_width = 1 # 不同列设置一个格间距
x.add_row(["113",cal_mean(X),cal_range(X),cal_var(X),cal_cut_mean(X),cal_frequency(X),cal_aad(X)]) # 添加两列数据
x.add_row(["114",cal_mean(Y), cal_range(Y),cal_var(Y),cal_cut_mean(Y),cal_frequency(Y),cal_aad(Y)])
print x   # 打印表格
