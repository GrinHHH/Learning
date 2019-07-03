# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import re
import requests
import pybase64
import os
# import fontforge

# fs = 1000
# lowcut = 100
# highcut = 300
# order = 1
# nyq = 0.5 * fs
# low = lowcut / nyq
# high = highcut / nyq
# b, a = signal.butter(order, [low, high], btype='band')


# dic = {'path':['wind/0306/x.txt','wind/030501/x.txt','wind/030502/x.txt']}
# for path in dic['path']:
#
#     data = np.loadtxt('wind/0306/x.txt')
#     # data = np.loadtxt('3-6/0.txt')
#     plt.plot(data)
#     plt.show()
#     plt.close()
# count = len(data)/2000

# data = np.loadtxt('数据/wind/震动传感器大风条件下测试数据/20190311（唐森风)/六号 大风/x.txt')
# count = 10
# for i in range(count):
#     y1 = np.fft.fft(data[(i)*2000:(i+1)*2000])[:1000]
#     fftx = np.linspace(0, 250, 999)
#     plt.plot(fftx,abs(y1[1:1000]))
#     # plt.savefig('image-fr/en/en_%d.jpg'%i)
#     plt.show()
#     plt.close()


# data = np.loadtxt('唐森/disposed/man/6.txt')
# plt.plot(data[0:100])
# plt.ylabel('value')
# plt.xlabel('point')
# plt.title('fool')
# plt.plot(data[100:200])
# plt.show()
# plt.close()
#
# t1 = np.arange(0.0, 1.0, 0.01)
# for n in [1, 2, 3, 4]:
#     plt.plot(t1, t1**n, label="n=%d"%(n,))
#
# plt.legend(('1','2','3'),loc='upper left', ncol=1,  shadow=False, fancybox=False)
# # leg.get_frame().set_alpha(0.5)
#
# plt.grid(True)
# plt.show()




