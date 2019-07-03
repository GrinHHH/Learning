# -*- coding: utf-8 -*-
import numpy as np      #导入np方便矩阵运算
import xlrd     #用于读取表格


def xls_tolist_unicode(xls):
    temp = []
    for i in range(90):
        temp.append(xls.row_values(i + 1))
    result = np.array(temp)
    result = np.delete(result, 0, axis=1)
    for i in range(90):
        if result[i][6] == 'yes':
            result[i][6] = 1
        else:
            result[i][6] = 0
    return result


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


def cal_gini(data):
    amount = len(data)
    type1 = 0
    type2 = 0
    for i in range(amount):
        if data[i][-1] == 0:
            type1 += 1
        else:
            type2 += 1
    gini = 1.0 - np.square(type1 * 1.0 / amount) - np.square(type2 * 1.0 / amount)
    return gini


def cal_split_gini(value, data, feature):   #计算最优gini的辅助函数，value为边界，data为输入数据，feature为属性在data中的列标
    amount_left = 0         #小于value的数量
    amount_right = 0        #大于value的数量
    right_type_0 = 0        #大于value类为0的数量
    right_type_1 = 0        #大于value类为1的数量
    left_type_0 = 0         #小于value类为0
    left_type_1 = 0         #小于value类为1
    for i in range(len(data)):      #遍历data行
        if data[i][feature] < value:        #该循环内部得到以上6个变量的值
            amount_left += 1
            if data[i][-1] == 0.0:
                left_type_0 += 1
            else:
                left_type_1 += 1
        else:
            amount_right += 1
            if data[i][-1] == 0:
                right_type_0 += 1
            else:
                right_type_1 += 1
    gini_left = 1.0 - np.square(left_type_0 * 1.0 / (left_type_0 + left_type_1)) - np.square(       #该部分为按照公式计算Gini的值
        left_type_1 * 1.0 / (left_type_0 + left_type_1))
    gini_right = 1.0 - np.square(right_type_0 * 1.0 / (right_type_0 + right_type_1)) - np.square(
        right_type_1 * 1.0 / (right_type_1 + right_type_0))
    gini = (amount_left * 1.0 / (amount_left + amount_right)) * gini_left + (
            amount_right * 1.0 / (amount_right + amount_left)) * gini_right
    return gini         #函数返回最终算得的Gini值


def divide_data(value, data, feature):      #定义函数切分数据为两部分，value为切分数据依据，feature为切分属性依据，data为带标签数据
    left = []       #存放小于value的数据集
    right = []      #存放大于value的数据集
    for rows in range(len(data)):      #遍历data行
        if data[i][feature] <= value:
            left.append(data[rows])        #在feature列小于value的分入左部分
        else:
            right.append(data[rows])       #大于的分入右部分
    left = np.array(left)       #左右均转化为np型list数据
    right = np.array(right)
    return left, right      #函数返回两个np的矩阵数据集


def cal_best_gini(feature, data):       #定义函数计算最符合节点分裂的gini值（属性值），feature为选择的属性，data为带标签数据
    if len(set([d[-1] for d in data])) == 1:
        return None        #如果数据已经为一类则不计算
    order = np.sort(data[:, feature])       #按feature给原始数据排序
    best_gini = 1       #存放结果的变量
    position = 0        #用于指出得到最小gini时属性值在顺序中的位置
    for rows in range(len(order) - 1):         #遍历数据行
        gini = cal_split_gini((order[rows] + order[rows + 1]) / 2.0, data, feature)       #调用函数计算gini
        if gini < best_gini:        #该部分用于得到最小的gini值
            best_gini = gini
            position = rows
    return order[position], best_gini    #函数返回gini值与取最小gini的属性值


def cut(data):      #定义函数作为剪枝的辅助函数，取节点中类别较多值的作为该节点的最终类别
                    #在最终应用中，考虑在gini小于某个值时调用该函数，防止过拟合，data为带标签数据
    type0 = 0       #存放标签数量
    type1 = 0
    for i in range(len(data)):      #遍历数据行，该循环用于得到所有标签的数量
        if data[i][-1]==0.0:
            type0 +=1
        else:
            type1 +=1
    if type0>type1:         #判断标签数量多少，多的则为最终的类别
        type = 0.0
    else:type = 1.0
    return type         #返回类别


def createClassifTree(data):        #定义函数训练决策树，data为带标签原始数据
    global start        #调用全局变量start，以保证固定顺序完成节点分裂
    a = set([d[-1] for d in data])      #集合用于存放数据中的类别
    if len(set([d[-1] for d in data])) == 1.0:  # 如果集合仅含一类，则节点为叶节点，函数直接返回类别
        start -= 1      #由于该部分迭代完毕，需要返回上一层的位置，全局变量也要返回上一侧的值
        if 1.0 in a:
            return 1.0
        else:
            return 0.0
    value,gini = cal_best_gini(divide_order[start], data)       #得到节点分裂的值
    # if gini <= threshold:         #用于剪枝，但由于不好确定gini的临界值，故暂时不进行操作
    #     return cut(data)
    left_child, right_child = divide_data(value, data, divide_order[start])     #左右节点存放按照value切分的数据
    classifTree = {}        #定义字典存放决策树模型
    classifTree['featIndex'] = feature_order[divide_order[start]]       #节点标签
    classifTree['value'] = value        #节点分裂的依据值
    start += 1      #加1表示进入下一层节点
    classifTree['leftChild'] = createClassifTree(left_child)    #迭代左分支，直至得到确定的分类
    start += 1      #在函数最初进行了现场的保护及返回，因此进入右分支时全局变量仍要加一
    classifTree['rightChild'] = createClassifTree(right_child)      #迭代右分支
    return classifTree      #返回决策树模型


def label_position(label): #该函数用于在预测时读取树模型中的属性，并返回属性在原始数据的列数，label为模型中读取到的属性名称
    position = 0        #存放最终的列数位置
    for i in range(len(feature_order)):         #遍历数据属性
        if feature_order[i] == label:           #找到与输入相同的属性列标
            position = i                        #赋值
    return position     #返回列标


def predict(tree, data):        #定义函数预测分类，tree为树模型，data为无标签测试数据
    feature = label_position(tree.items()[1][1])        #读取根节点标签的列标
    value = tree.items()[2][1]      #读取根节点分裂数值依据
    if data[feature] <= value:      #小于value则进入左分支
        if isinstance(tree.items()[0][1],float):        #左分支中，若为float则为叶节点，直接返回结果
            return tree.items()[0][1]
        else:       #若不是float类型，则为词典形式保存的树分支，迭代进入子树
            result = predict(tree.items()[0][1],data)
            return  result      #返回子树的分类结果
    else:       #大于value将进入右分支
        if isinstance(tree.items()[3][1],float):        #同上，float为叶节点类别
            return tree.items()[3][1]
        else:       #不是float则进入子树迭代
            result = predict(tree.items()[3][1], data)
            return  result      #返回子树分类结果


start = 0       #定义全局变量，与分裂顺序一同控制决策树的节点属性
divide_order = [1,5,4,0,3]      #定义矩阵存放决策树顺序，可随意修改
# feature_order = ['max','min','mean','range','var','std']    #与表格数据的属性对应，用于节点模型保存
feature_order = [0,1,2,3,4,5]
threshold = 0.01        #剪枝边界，由于没有界定经验，故不进行剪枝
result = []         #用于存放所有测试集的结果
workbook_train = xlrd.open_workbook(r'e:/Learning/mining/Data_113.xls')        #读取113表格
workbook_test = xlrd.open_workbook(r'e:/Learning/mining/Data_114.xls')     #读取114表格
sheet_train = workbook_train.sheet_by_name('sheet1')        #读取113表格第一页
sheet_test = workbook_test.sheet_by_name('sheet1')          #读取114表格第一页
x = xls_tolist_python(sheet_train)      #转化为np矩阵形式
y = xls_tolist_python(sheet_test)       #转化为np矩阵形式

tree_mod = createClassifTree(x)   #进行决策树训练，模型保存在变量中
print 'Tree mod:',str(tree_mod).replace(' ','').replace('{','').replace('}','') #打印树模型
for i in range(len(y)):         #遍历测试集原始数据，预测各个数据值，并保存在result中
    result.append(predict(tree_mod, y[i]))
result = np.array(result)       #转化为np矩阵
TP = 0.         #以下定义度量值
TN = 0.
FP = 0.
FN = 0.
for i in range(len(y)):         #循环得到各个度量值的基本算子的值
    if y[i][-1] == 1.0:
        if result[i] == 1.0:
            TP += 1
        else:
            FN += 1
    else:
        if result[i] == 0.0:
            TN += 1
        else:
            FP += 1
Accuracy = (TP+TN)/(TP+TN+FP+FN)    #精确度
TPR = TP/(TP+FN)    #真正率
TNR = TN/(TN+FP)    #真负率
FPR = FP/(TN+FP)    #假正率
FNR = FN/(TP+FN)    #假负率
Precision = TP/(TP+FP)  #精度
Recall = TP/(TP+FN)     #召回率
F_measure = 2*TP/(2*TP+TP+FN)   #F1度量
print 'TPR =', TPR      #打印各个度量值
print 'TNR =', TNR
print 'FPR =', FPR
print 'Precision =', Precision
print 'Recall =', Recall
print 'Accuracy =', Accuracy
print 'F_measure =', F_measure
