# -*- coding: utf-8 -*-
#########################################################################

"""
Created on Fri Jan 06 10:08:42 2017

@author: Yuyangyou

代码功能描述：（1）读取Sharp_waves文件，
              （2）采用巴特沃斯滤波器，进行60-240Hz滤波
              （3）画图
              （4）....

"""
#####################################################################


import numpy as np
from scipy import signal
import math
import matplotlib
import matplotlib.pylab as plt
#from numpy import *
import os
import io
import gc   #gc模块提供一个接口给开发者设置垃圾回收的选项
import time
import collections    #这个模块实现了一些很好的数据结构
import xlwt

#读取文件第一列，保存在s1列表中
###########################################################################################################
start = 113 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
N = 1                                                      #设立变量N，作为循环读取文件的增量
for e in range(start,start+N):                            #循环2次，读取113&114两个文件
    data = open(r'c:/data/20151026_%d.txt'% (e)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
    data = data.split( )                                  #以空格为分隔符，返回数值列表data
    data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型

    s1 = data[0:45000*4:4]                          #list切片L[n1:n2:n3]  n1代表开始元素下标；n2代表结束元素下标
                                                    #n3代表切片步长，可以不提供，默认值是1，步长值不能为0
####################################################################################################################


#滤波
##################################################################################################################
    fs = 3000                                           #设立频率变量fs
    lowcut = 1
    highcut = 30
    order = 2                                           #设立滤波器阶次变量
    nyq = 0.5*fs                                        #设立采样频率变量nyq，采样频率=fs/2。
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter(order,[low,high],btype='band') #设计巴特沃斯带通滤波器 “band”
    s1_filter1 = signal.lfilter(b,a,s1)                 #将s1带入滤波器，滤波结果保存在s1_filter1中
###################################################################################################################





#划分分析粒度并保存文件——（1）分析粒度不够精细，反复试验寻找合理分析粒度
##############################################################################################################
    count_temp=s1_filter1                           #设置临时变量count_temp，复制滤波后array，防止操作改变原始数据
    for count_loop in range (90):                   #设置变量count_loop，用于记数，循环90次
        count_500=count_temp[:500]                  #设置变量count_500,对count_temp array进行切片，从0切到500，赋值给count_500
        np.savetxt(r'c:/data/20151026_%d.txt' % (count_loop), count_500)    #将切好的500长的numpy array 保存为txt文件

        # fig_count_loop = plt.figure()               # 创建画图对象，开始画图
        # plt.plot(count_500, color='r')              #画count_500序列，红色
        # plt.savefig(r'c:/data/20151026_%d.png' % (count_loop))  # 保存图像，设定保存路径并统一命名，%d处，十进制替换为count_loop值
        # plt.close('all')                            #释放绘图资源
        count_temp = count_temp[500:]               #切片，从500至结尾，赋值给count_temp


#################################################################################################################



#画图
###################################################################################################################
    fig_review = plt.figure()                             #创建画图对象，开始画图

    plt.plot(s1_filter1,color='r')                         #s1_filter1，红色
    plt.savefig(r'c:/data/113.png' )                     #保存图像
    plt.close('all')                                       #释放绘图资源


##################################################################################################################



#计算6种属性
###########################################################################################################

for cal_loop in range(90):                                               #循环90次
    cal_data = open(r'c:/data/20151026_%d.txt'% (cal_loop)).read()     #读取长度为500的序列文件
    cal_data = cal_data.split()                                          #去除多余字符
    cal_data = [float(s) for s in cal_data]                              #强制类型转换

    #计算均值并写入cal_data_mean.txt文件
    ###################################################################################################
    cal_data_mean=sum(cal_data)/500                                    #计算均值
    data_mean = open(r'c:/data/20151026_113_mean.txt', 'a+')        #'a+'将均值以追加形式写入文件，文件不存在即创建
    data_mean.write(str(cal_data_mean)+'\n')                           #装换为字符串，写入文件并换行
    data_mean.close()                                                  #释放文件读写资源
    ###################################################################################################

    #计算最大值写入cal_data_max.txt文件
    ######################################################################################################
    cal_data_max=max(cal_data)                                   #计算最大值
    data_max = open(r'c:/data/20151026_113_max.txt', 'a+')    #'a+'将最大值以追加形式写入文件，文件不存在即创建
    data_max.write(str(cal_data_max) + '\n')                     #装换为字符串，写入文件并换行
    data_max.close()                                             #释放文件读写资源
    ###################################################################################################################


    #计算最小值写入cal_data_min.txt文件
    ######################################################################################################
    cal_data_min=min(cal_data)                                   #计算最小值
    data_min = open(r'c:/data/20151026_113_min.txt', 'a+')    #'a+'将最大值以追加形式写入文件，文件不存在即创建
    data_min.write(str(cal_data_min) + '\n')                     #装换为字符串，写入文件并换行
    data_min.close()                                             #释放文件读写资源
    ###################################################################################################################


    #计算极差写入cal_data_range.txt文件
    ######################################################################################################
    cal_data_range=max(cal_data)-min(cal_data)                     #计算极差
    data_range = open(r'c:/data/20151026_113_range.txt', 'a+')  #'a+'将最大值以追加形式写入文件，文件不存在即创建
    data_range.write(str(cal_data_range) + '\n')                   # 装换为字符串，写入文件并换行
    data_range.close()                                             #释放文件读写资源
    ###################################################################################################################


    #计算标准差写入cal_data_var.txt文件
    ######################################################################################################
    var_ini = 0                                                     # 给var_ini 赋初值
    for var_loop in cal_data:                                       # 遍历cal_data中的每一个数
        var_ini += (var_loop - (sum(cal_data)/500)) ** 2            # 计算cal_data中的元素与均值的平方和
    cal_data_var = float(var_ini) / 500                             # 计算方差
    data_var = open(r'c:/data/20151026_113_var.txt', 'a+')        # 'a+'将最大值以追加形式写入文件，文件不存在即创建
    data_var.write(str(cal_data_var) + '\n')                        # 装换为字符串，写入文件并换行
    data_var.close()                                                #释放文件读写资源
    ###################################################################################################################


    #计算标准差写入cal_data_std.txt文件
    ######################################################################################################
    cal_data_std = math.sqrt(cal_data_var)                      # 对方差开根号
    data_std = open(r'c:/data/20151026_113_std.txt', 'a+')    #'a+'将最大值以追加形式写入文件，文件不存在即创建
    data_std.write(str(cal_data_var) + '\n')                    # 装换为字符串，写入文件并换行
    data_std.close()                                             #释放文件读写资源
    ###################################################################################################################



#赋标签，判断极差是否大于0.05，大于为yes，小于为no。（2）——标签信息不充分，设计属性作为判断条件，给出准确标签
########################################################################################################################
range_value = open(r'c:/data/20151026_113_range.txt').read()     #设立range_value列表变量.read读文件
range_value = range_value.split( )                                  #以空格为分隔符，返回数值列表range_value
range_value = [float(val_range) for val_range in range_value]       #强制类型转换

max_value = open(r'c:/data/20151026_113_max.txt').read()     #设立max_value列表变量.read读文件
max_value = max_value.split( )                                  #以空格为分隔符，返回数值列表max_value
max_value = [float(val_max) for val_max in max_value]           #强制类型转换

min_value = open(r'c:/data/20151026_113_min.txt').read()     #设立min_value列表变量.read读文件
min_value = min_value.split( )                                  #以空格为分隔符，返回数值列表min_value
min_value = [float(val_min) for val_min in min_value]           #强制类型转换

mean_value = open(r'c:/data/20151026_113_mean.txt').read()     #设立mean_value列表变量.read读文件
mean_value = mean_value.split( )                                  #以空格为分隔符，返回数值列表mean_value
mean_value = [float(val_mean) for val_mean in mean_value]           #强制类型转换

std_value = open(r'c:/data/20151026_113_std.txt').read()     #设立std_value列表变量.read读文件
std_value = std_value.split( )                                  #以空格为分隔符，返回数值列表std_value
std_value = [float(val_std) for val_std in std_value]           #强制类型转换

var_value = open(r'c:/data/20151026_113_var.txt').read()     #设立var_value列表变量.read读文件
var_value = var_value.split( )                                  #以空格为分隔符，返回数值列表var_value
var_value = [float(val_var) for val_var in var_value]           #强制类型转换

file_excel = xlwt.Workbook()                                    # 新建xcel文件
file_excel_sheet1 = file_excel.add_sheet(u'sheet1',cell_overwrite_ok=True)  #添加工作表sheet


for loop in range(90):                                          #循环90次，写入excel文件
    row=loop+1                                                     #设置变量row，作为行标号，初始值为loop+1
    column=1                                                       #设置变量column，作为列标号，初始值为1
    file_excel_sheet1.write(row, column, max_value[loop])           #写入第row行，第column列，写入值为max_value[loop]
    column+=1                                                       #列值自增运算，自加1
    file_excel_sheet1.write(row, column, min_value[loop])           #写入第row行，第column列，写入值为min_value[loop]
    column+=1                                                       #列值自增运算，自加1
    file_excel_sheet1.write(row, column, mean_value[loop])          #写入第row行，第column列，写入值为mean_value[loop]
    column+=1                                                       #列值自增运算，自加1
    file_excel_sheet1.write(row, column, range_value[loop])         #写入第row行，第column列，写入值为range_value[loop]
    column+=1                                                       #列值自增运算，自加1
    file_excel_sheet1.write(row, column, var_value[loop])           #写入第row行，第column列，写入值为var_value[loop]
    column+=1                                                       #列值自增运算，自加1
    file_excel_sheet1.write(row, column, std_value[loop])           #写入第row行，第column列，写入值为std_value[loop]

    column+=1                                                       #列值自增运算，自加1
    if range_value[loop]>0.05:                                      #条件判断极差是否大于0.05，如大于
        file_excel_sheet1.write(row, column, 'yes')                 #写入写入第row行，第column列，写入值为标签yes
    else:                                                           #条件判断极差是否大于0.05，如小于
        file_excel_sheet1.write(row, column, 'no')                  #写入写入第row行，第column列，写入值为标签no

file_excel.save(r'c:\data\Data.xls')                               # 特征写入完毕，保存数据为Data.xls文件，路径为c:\data



