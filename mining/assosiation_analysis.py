# -*- coding: utf-8 -*-
import numpy as np      #导入np方便矩阵运算
import xlrd     #用于读取表格


def xls_tolist_python(xls):     #定义函数将读取的表格数据转换为list数据
    temp = [[0 for i in range(7)] for j in range(90)]       #建立空list，存放接下来的表格数据
    for i in range(90):     #遍历表格数据行数
        for j in range(7):  #便利表格列数
            temp[i][j] = xls.cell(i + 1, j + 1).value   #读取数据，赋值
    cols = xls.col_values(7)        #最后一列为标签
    for i in range(len(temp)):      #遍历temp行
        if cols[i + 1] == 'yes':        #标签为yes则赋值1，no则赋值0
            temp[i][6] = 1
        else:
            temp[i][6] = 0
    result = np.array(temp)     #最终结果为python的list，将其转化为np的list，方便处理
    return result       #返回list数据


def list_to_interval(l):        #表格数据映射至0-1，并划分为5个区间
    temp = [[0 for i in range(6)] for j in range(90)] #建立中间变量存放结果区间
    for i in range(6):      #按列读表
        max_num = np.max(l[:, i])   #取该列最大值
        min_num = np.min(l[:, i])   #取最小值
        for j in range(90):         #遍历列
            temp[j][i] = (l[j][i] + abs(min_num)) / (max_num + abs(min_num)) #映射至0-1
            if temp[j][i] < 0.2:    #分为5个区间，并赋值0，1，2，3，4
                temp[j][i] = 0
            elif (temp[j][i] >= 0.2) & (temp[j][i] < 0.4):
                temp[j][i] = 1
            elif (temp[j][i] >= 0.4) & (temp[j][i] < 0.6):
                temp[j][i] = 2
            elif (temp[j][i] >= 0.6) & (temp[j][i] < 0.8):
                temp[j][i] = 3
            else:
                temp[j][i] = 4
    temp = np.array(temp) # 转换为np矩阵形式
    return temp     #返回区间数据


def cal_support(data, prop, value): #计算支持度，输入为区间数据，属性数组及对应的数值数组
    num_length = len(data) #得到区间数据长度
    count = 0.      #支持计数
    cycle_time = len(prop) #需要比较的数据个数
    for rows in range(len(data)): #遍历行
        for num in range(cycle_time): #每行的数据对应个数
            if data[rows][prop[num]] != value[num]: #若有一个不相等，则停止比较
                break
            elif num == cycle_time-1:   #若到最后都相等，则计数加一
                count += 1
    return count/num_length  #返回支持度


def gen_set(temp,minsup,data_set,order):    #生成项集
    set_sup = [] #存放支持度
    fq_set = [[[0,0]for i in range(order)]] #存放最终生成的项集
    fq_set_temp = [[0,0]for i in range(order)] # 临时变量，用于存放生成的单个项集
    prop = [0 for i in range(order)] #存放需要比对的属性
    num = [0 for i in range(order)] #存放需要比对的值
    for set_len in range(len(data_set)-1): #遍历上一个项集
        for another in range(len(data_set)-set_len-1): #两两项集对比
            if order<=2: #2项集没有相同项，故分开计算
                if data_set[set_len][0][0]!=data_set[set_len+another+1][0][0]:  #属性类别不同则合并
                    prop[0] = data_set[set_len][0][0]       #新增属性
                    num[0] = data_set[set_len][0][1]        #新增对应属性的值
                    prop[1] = data_set[set_len + another + 1][0][0]
                    num[1] = data_set[set_len + another + 1][0][1]
                    if cal_support(temp, prop, num) > minsup:   #支持度若大于minsup
                        set_sup = np.append(set_sup, cal_support(temp, prop, num)) #保存支持度
                        fq_set = np.concatenate((fq_set, [[data_set[set_len][0], data_set[set_len + another + 1][0]]]))#保存该项
            else:   #三项集及以上
                bug = [[0, 0] for i in range(order - 2)] #存放前几个相同的项，这里总是出bug
                k=0     #计数相同项的个数
                for same_element in range(order-2): #取前（项数-2）个项对比
                    if np.array_equal(data_set[set_len][same_element],data_set[set_len+another+1][same_element]):#相等则计数+1
                        k+=1
                        if (same_element == (order-3))&(k==(order-2)):#计数达到要求时合并项集，生成新项集
                            if data_set[set_len][same_element+1][0]!=data_set[set_len+another+1][same_element+1][0]:#合并的项类别必须不同
                                for element_position in range(order - 2): #新项集赋值
                                    prop[element_position] = data_set[set_len][element_position][0]
                                    num[element_position] = data_set[set_len][element_position][1]
                                    bug[element_position] = data_set[set_len][element_position]
                                prop[order - 2] = data_set[set_len][order - 2][0]
                                num[order - 2] = data_set[set_len][order - 2][1]
                                prop[order - 1] = data_set[set_len + another + 1][order - 2][0]
                                num[order - 1] = data_set[set_len + another + 1][order - 2][1]
                                if cal_support(temp, prop, num) > minsup: #计算支持度，大于则保存
                                    fq_set_temp[order - 2] = data_set[set_len][order - 2] #将新增的项付给变量的后几个位置
                                    fq_set_temp[order - 1] = data_set[set_len + another + 1][order - 2]
                                    for moo in range(order - 2): #将相同项赋值给单个项的前几个位置
                                        fq_set_temp[moo] = bug[moo]
                                    set_sup = np.append(set_sup, cal_support(temp, prop, num))#保存
                                    fq_set = np.concatenate((fq_set, [fq_set_temp]))
    return set_sup,fq_set[1:len(fq_set)] #返回支持度数组及项集




min_sup = 0.2       #设定最小支持度与置信度
min_conf = 0.5
workbook = xlrd.open_workbook(r'e:/Learning/Data_113.xls')        #读取113表格
sheet_train = workbook.sheet_by_name('sheet1')        #读取113表格第一页
x = xls_tolist_python(sheet_train)      #转化为np矩阵形式
interval = list_to_interval(x)      #转化为区间数据
set_1_sup = []      #定义一项集支持度
first = [[[0,0]]]   #一项集初始化
for data_prop in range(5):      #生成一项集，由于表格第六列数据与第五列相同，故舍弃第六列
    for data_value in range(5):     #五个区间
        sup = cal_support(interval,[data_prop],[data_value])    #计算支持度
        if sup>min_sup: #大于minsup则保存
            set_1_sup = np.append(set_1_sup,sup)    #对每一个项的支持度进行保存
            first=np.concatenate((first,[[[data_prop,data_value]]]))       #将每一个项写入一项集变量

set_1 = first[1:len(first)]         #第一行为冗余，删除
set_2_sup,set_2 = gen_set(interval,min_sup,set_1,2)     #计算二项集及其支持度
set_3_sup,set_3 = gen_set(interval,min_sup,set_2,3)     #计算三项集及其支持度
set_4_sup,set_4 = gen_set(interval,min_sup,set_3,4)     #计算四项集及其支持度
set_5_sup,set_5 = gen_set(interval,min_sup,set_4,5)     #计算五项集及其支持度
print set_1     #打印频繁项
print set_2
print set_3
print set_4
print set_5
# print np.shape(set_1) #打印频繁项的个数
# print np.shape(set_2)
# print np.shape(set_3)
# print np.shape(set_4)
# print np.shape(set_5)
