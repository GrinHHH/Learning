import numpy as np      # 导入np方便矩阵运算
import xlrd     # 用于读取表格
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def xls_tolist_python(xls,property_num,data_len):     # 定义函数将读取的表格数据转换为list数据，输入表格，列数，行数
    temp = [[0 for i in range(property_num)] for j in range(data_len)]       # 建立空list，存放接下来的表格数据
    for rows in range(data_len):     # 遍历表格数据行数
        for cols in range(property_num):  # 便利表格列数
            temp[rows][cols] = xls.cell(rows, cols).value   # 读取数据，赋值
    xls_result = np.array(temp)     # 最终结果为python的list，将其转化为np的list，方便处理
    return xls_result       # 返回list数据


def delete_same_element(data):
    temp = []
    temp.append(data[0])
    for rows in range(1, len(data)):
        if data[rows] != data[rows-1]:
            temp.append(data[rows])
    temp = np.array(temp)
    return temp


def cal_split_gini(value, data, cols):   # 辅助函数，value为边界，data为输入数据，feature为属性在data中的列标
    amount_left = 0         # 小于value的数量
    amount_right = 0        # 大于value的数量
    right_type_0 = 0        # 大于value类为0的数量
    right_type_1 = 0        # 大于value类为1的数量
    left_type_0 = 0         # 小于value类为0
    left_type_1 = 0         # 小于value类为1
    for rows in range(len(data)):
        if data[rows][cols] <= value:        # 该循环内部得到以上6个变量的值
            amount_left += 1
            if data[rows][-1] == 0.0:
                left_type_0 += 1
            else:
                left_type_1 += 1
        else:
            amount_right += 1
            if data[rows][-1] == 0:
                right_type_0 += 1
            else:
                right_type_1 += 1
    gini_left = 1.0 - np.square(left_type_0 * 1.0 / (left_type_0 + left_type_1)) - np.square(       # 该部分为按照公式计算Gini的值
        left_type_1 * 1.0 / (left_type_0 + left_type_1))
    gini_right = 1.0 - np.square(right_type_0 * 1.0 / (right_type_0 + right_type_1)) - np.square(
        right_type_1 * 1.0 / (right_type_1 + right_type_0))
    gini = (amount_left * 1.0 / (amount_left + amount_right)) * gini_left + (
            amount_right * 1.0 / (amount_right + amount_left)) * gini_right
    return gini


def divide_data(value, data, feature):      # 定义函数切分数据为两部分，value为切分数据依据，feature为切分属性依据，data为带标签数据
    left = []       # 存放小于value的数据集
    right = []      # 存放大于value的数据集
    for rows in range(len(data)):      # 遍历data行
        if data[rows][feature] <= value:
            left.append(data[rows])        # 在feature列小于value的分入左部分
        else:
            right.append(data[rows])       # 大于的分入右部分
    left = np.array(left)       # 左右均转化为np型list数据
    right = np.array(right)
    return left, right      # 函数返回两个np的矩阵数据集


def cal_best_gini(data):       #定义函数计算最符合节点分裂的gini值（属性值），feature为选择的属性，data为带标签数据
    gini_final = 1
    value = 0
    feature_position = 0
    for cols in range(len(feature)):
        # if len(set([d[-1] for d in data])) == 1:
        #     return None  # 如果数据已经为一类则不计算
        order = np.sort(data[:, cols])  # 按feature给原始数据排序
        order = delete_same_element(order)
        best_gini = 1  # 存放结果的变量
        position = 0  # 用于指出得到最小gini时属性值在顺序中的位置
        for rows in range(len(order) - 1):  # 遍历数据行
            gini = cal_split_gini((order[rows] + order[rows + 1]) / 2.0, data, cols)  # 调用函数计算gini
            if gini < best_gini:  # 该部分用于得到最小的gini值
                best_gini = gini
                position = rows
        if best_gini < gini_final:
            gini_final = best_gini
            value = order[position]
            feature_position = cols
    return value, gini_final, feature_position    #函数返回gini值与取最小gini的属性值


def cut(data):      #定义函数作为剪枝的辅助函数，取节点中类别较多值的作为该节点的最终类别
                    #在最终应用中，考虑在gini小于某个值时调用该函数，防止过拟合，data为带标签数据
    type1 = 0       #存放标签数量
    type2 = 0
    for i in range(len(data)):      #遍历数据行，该循环用于得到所有标签的数量
        if data[i][-1]==1.0:
            type1 +=1
        else:
            type2 +=1
    if type1>type2:         #判断标签数量多少，多的则为最终的类别
        type = 1.0
    else:type = 2.0
    return type         #返回类别


def createClassifTree(data):        #定义函数训练决策树，data为带标签原始数据
    a = set([d[-1] for d in data])      #集合用于存放数据中的类别
    # if count>6:
    #     return cut(data)
    if len(a) == 1:  # 如果集合仅含一类，则节点为叶节点，函数直接返回类别
        if 1.0 in a:
            return 1.0
        elif 2.0 in a:
            return 2.0
        else:
            return 0.0
    value, gini, cols = cal_best_gini(data)       #得到节点分裂的值
    # if gini <= threshold:         #用于剪枝，但由于不好确定gini的临界值，故暂时不进行操作
    #     return cut(data)
    left_child, right_child = divide_data(value, data, cols)     #左右节点存放按照value切分的数据
    # if len(left_child)==0 or len(right_child)==0:
    #     print data
    classifTree = {}

    # count+=1
    classifTree['leftChild'] = createClassifTree(left_child)    #迭代左分支，直至得到确定的分类
    classifTree['featIndex'] = feature[cols]  # 节点标签
    classifTree['value'] = value  # 节点分裂的依据值
    classifTree['rightChild'] = createClassifTree(right_child)      #迭代右分支
    # count-=1
    return classifTree      #返回决策树模型


def label_position(label):
    position = 0        # 存放最终的列数位置
    for element in range(len(feature)):
        if feature[element] == label:           # 找到与输入相同的属性列标
            position = element
    return position     # 返回列标


def predict(tree, data):        # 定义函数预测分类，tree为树模型，data为无标签测试数据
    temp = list(tree.items())
    cols = label_position(temp[1][1])        # 读取根节点标签的列标
    value = temp[2][1]      # 读取根节点分裂数值依据
    if data[cols] <= value:      # 小于value则进入左分支
        if isinstance(temp[0][1], float):        # 左分支中，若为float则为叶节点，直接返回结果
            return temp[0][1]
        else:       # 若不是float类型，则为词典形式保存的树分支，迭代进入子树
            pre_result = predict(temp[0][1], data)
            return pre_result      # 返回子树的分类结果
    else:       # 大于value将进入右分支
        if isinstance(temp[3][1], float):        # 同上，float为叶节点类别
            return temp[3][1]
        else:       # 不是float则进入子树迭代
            pre_result = predict(temp[3][1], data)
            return pre_result      # 返回子树分类结果


# feature_order = ['max','min','mean','range','var','std']    # 与表格数据的属性对应，用于节点模型保存

# 从这里开始
# 目前特征有5个，就01234代表了，这里就单纯地是自然数排列代表每个特征
# 之前程序写的像一坨屎，这里就不好改了= =见谅
feature = [0,1,2,3,4]
# threshold = 0.01        # 剪枝边界，由于没有界定经验，故不进行剪枝
result = []         # 用于存放所有测试集的结果
result_2=[]

# 整个下面一块用来读取之前生成的excel，函数输入，后两个为列数和行数，
# 列数必须和excel相同，行数小于等于excel，相当于取多少样本
workbook1 = xlrd.open_workbook(r'唐森/car.xls')
sheet_train1 = workbook1.sheet_by_name('sheet1')
x1 = xls_tolist_python(sheet_train1, 6, 172)

workbook2 = xlrd.open_workbook(r'唐森/man.xls')
sheet_train1 = workbook2.sheet_by_name('sheet1')
x2 = xls_tolist_python(sheet_train1, 6, 242)

workbook3 = xlrd.open_workbook(r'唐森/en2.xls')
sheet_train1 = workbook3.sheet_by_name('sheet1')
x3 = xls_tolist_python(sheet_train1, 6, 236)

workbook4 = xlrd.open_workbook(r'唐森/en2.xls')
sheet_train1 = workbook4.sheet_by_name('sheet1')
x4 = xls_tolist_python(sheet_train1, 6, 110)
# 到这里数据加载完毕
# 首先把车的标签置为1，做一次二分类
# 即先区分有没有目标,可以理解为目标检测部分
for row in range(len(x1)):
    x1[row][-1] = 1
# 生成数据集
data_set1 = np.concatenate((x1, x2, x3,x4), axis=0)
# 再把标签变为0，车0人1做一次二分类
for row in range(len(x1)):
    x1[row][-1] = 0
# 这里只用人和车的数据
data_set2 = np.concatenate((x1, x2), axis=0)
# 下面为数据集分割
train_set_1, test_set_1 = train_test_split(data_set1, test_size=0.2)
train_set_2, test_set_2 = train_test_split(data_set2, test_size=0.2)

# 生成决策树模型
tree_mod_1 = createClassifTree(train_set_1)
tree_mod_2 = createClassifTree(train_set_2)
# 模型打印,之后粘贴到单片机程序中,全局变量里有两个部分
print('Tree mod_1:', str(tree_mod_1).replace(' ', '').replace('{', '').replace('}', ''))
print('Tree mod_2:', str(tree_mod_2).replace(' ', '').replace('{', '').replace('}', ''))
# print 'Tree mod_1:', str(tree_mod_1).replace(' ', '').replace('{', '').replace('}', '')
# print 'Tree mod_2:', str(tree_mod_2).replace(' ', '').replace('{', '').replace('}', '')

# 下面就是测试了,输出两个混淆矩阵,看模型的分类效果,第一个是目标检测,第二个是人车
for i in range(len(test_set_1)):
    result.append(predict(tree_mod_1, test_set_1[i][0:7]))
result = np.array(result)
for i in range(len(test_set_2)):
    result_2.append(predict(tree_mod_2, test_set_2[i][0:7]))
result_2 = np.array(result_2)
Matrix = confusion_matrix(test_set_1[:, -1], result)
Matrix_2 = confusion_matrix(test_set_2[:, -1], result_2)
print(Matrix)
print(Matrix_2)

