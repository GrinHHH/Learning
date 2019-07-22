# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
# from scipy import signal
# import re
# import requests
# import pybase64
# import os
# import fontforge

# fs = 1000
# lowcut = 100
# highcut = 300
# order = 1
# nyq = 0.5 * fs
# low = lowcut / nyq
# high = highcut / nyq
# b, a = signal.butter(order, [low, high], btype='band')


# data = np.loadtxt('数据/wind/震动传感器大风条件下测试数据/20190311（唐森风)/六号 大风/x.txt')
# count = 10
# for i in range(count):
#     y1 = np.fft.fft(data[(i)*2000:(i+1)*2000])[:1000]
#     fftx = np.linspace(0, 250, 999)
#     plt.plot(fftx,abs(y1[1:1000]))
#     # plt.savefig('image-fr/en/en_%d.jpg'%i)
#     plt.show()
#     plt.close()
for i in range(2):

    data = np.loadtxt('唐森/2019-7-1/%d.txt'%i)
    plt.subplot(2,1,i+1)
    plt.plot(data)
plt.show()




