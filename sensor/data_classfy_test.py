# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal

class_type = 0
class_boundary = 320
other_boundary = 40
Threshold = 0.2
white_space = 100
fs = 1000  # 设立频率变量fs
lowcut = 5
highcut = 400
order = 1  # 设立滤波器阶次变量
nyq = 0.5 * fs  # 设立采样频率变量nyq，采样频率=fs/2。
low = lowcut / nyq
high = highcut / nyq
#num_list = np.loadtxt("new_data/human/4/01-14-34-24.txt")
#num_list = np.loadtxt("new_data/human/4/01-14-35-29.txt")
num_list = np.loadtxt("0.txt")
b, a = signal.butter(order, [low, high], btype='bandpass')  # 设计巴特沃斯带通滤波器 “band”
s1_filter1 = signal.lfilter(b, a, num_list)
x = s1_filter1[10000:20000]


def cal_white_num(data):
    num = 0

    for value in range(len(data)-white_space):
        if data[value] == 1:
            flag = 0
            for position in range(white_space):
                if data[value+position+1] == 1:
                    flag += 1
            if flag == 0:
                num += 1
                value += white_space
    return num


for i in range(len(x)): # 二值化
    if x[i] < Threshold:
        x[i] = 0
    else:
        x[i] = 1

for i in range(len(x) / 500): #去单个噪点
    sum = 0
    for j in range(500):
        sum = sum + x[j + i * 500]
    if sum <= 5:
        for k in range(500):
            x[k + i * 500] = 0

for i in range(len(x)): #开始判断
    space = 0
    if x[i] == 1:
        sum = 0
        for j in range(2000):
            sum = sum + x[i+j]
        print sum
        if sum <= other_boundary:
            class_type = 0
            print 'Object: Others'
        else:
            if sum >= class_boundary:
                class_type=2
                print 'Object: Car'
            else:
                for j in range(len(x)):
                    if x[j] == 1:
                        space = cal_white_num(x[j:j+2000])
                        print space
                        break
                if space>=6:
                    class_type = 1
                    print 'Object: Man'
                else:
                    class_type = 2
                    print 'Object: Car'
        break
