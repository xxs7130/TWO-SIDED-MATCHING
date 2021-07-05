# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:54:11 2020

@author: NANXIANG
"""

# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import random
import copy
import numpy as np

# In[]
# random.seed(1024)
# A 100个， 属性4个；  B 120个， 属性3个
a_len = 500
b_len = 500
# nind= 50 #种群规模
a_attri_num = 4
b_attri_num = 3

# In[]
wa = [0.263, 0.252, 0.219, 0.267]
wb = [0.288, 0.323, 0.389]

# In[]

# random.seed(1024)

ft1 = open("generante_hesitant_fuzzy_set_a.txt", "w+")

ft2 = open("generante_hesitant_fuzzy_set_b.txt", "w+")

ft3 = open("generante_hesitant_fuzzy_set_a_expand.txt", "w+")

ft4 = open("generante_hesitant_fuzzy_set_b_expand.txt", "w+")

# 产生单个犹豫模糊元素
def generate_h():
    a = []
    lh = random.randint(1, 4)
    for i in range(lh):
        a.append(round(random.random(), 1))
        a.sort()
    return a


# 犹豫模糊元素扩充到指定长度
# 传入元素和指定长度
def list_expand(a, lh):
    max_a = max(a)  # 最大值
    lh_a = len(a)  # a的长度
    b = copy.deepcopy(a)
    if (lh - lh_a != 0):
        for i in range(lh - lh_a):
            b.append(max_a)
    return b


# 求两个犹豫模糊元素之间的偏差度
def Distance(h1, h2):
    len_h = len(h1)
    temp_total = 0
    for i in range(len_h):
        temp1 = abs(h1[i] - h2[i])
        temp2 = temp1 ** 2
        temp_total = temp_total + temp2
    return (temp_total / len_h) ** 0.5



a_expend = []  # 扩充后的犹豫模糊矩阵
b_expend = []  # 扩充后的犹豫模糊矩阵

# a原始犹豫模糊矩阵与扩充矩阵
for i in range(a_len * a_attri_num):  # 100a*4 属性
    a_temp = []
    for j in range(b_len):  # 120b
        # 每一单元格的犹豫模糊元素
        temp_h = generate_h()
        h = list_expand(temp_h, 4)
        ft1.write('{')
        ft1.writelines(str(temp_h))
        ft1.write('}')
        ft1.write('*')

        ft3.write('{')
        ft3.writelines(str(h))
        a_temp.append(h)
        ft3.write('}')
        ft3.write('*')

    ft1.writelines("\n")
    ft3.writelines("\n")
    a_expend.append(a_temp)

ft1.close()
ft3.close()

# a原始犹豫模糊矩阵
for i in range(b_len * b_attri_num):  # b120*3
    b_temp = []
    for j in range(a_len):  # a100
        # 每一单元格的犹豫模糊元素
        temp_h = generate_h()
        h = list_expand(temp_h, 4)

        ft2.write('{')
        ft2.writelines(str(temp_h))
        ft2.write('}')
        ft2.write('*')

        ft4.write('{')
        ft4.writelines(str(h))
        b_temp.append(h)
        ft4.write('}')
        ft4.write('*')
    ft2.writelines("\n")
    ft4.writelines("\n")
    b_expend.append(b_temp)
ft2.close()
ft4.close()

# In[] 最大偏差法
# A对B评价的Dj
DAj = []

for attri in a_expend:  # attri是a_expend的每行，包含120个元素
    d_temp = 0
    for i in range(b_len):
        for j in range(b_len):
            d_temp = d_temp + Distance(attri[i], attri[j])  # 相对于其他所有主体的偏差
    DAj.append(d_temp)

# B对A评价的Dj
DBj = []

for attri in b_expend:
    d_temp = 0
    for i in range(a_len):
        for j in range(a_len):
            d_temp = d_temp + Distance(attri[i], attri[j])
    DBj.append(d_temp)




# In[] 犹豫模糊加权平均算子
# 传入b对a的评价
def GHFWA_btoa(evalution_list):
    lamda = 1
    GHFWA = []
    for h1 in evalution_list[0]:
        for h2 in evalution_list[1]:
            for h3 in evalution_list[2]:
                for h4 in evalution_list[3]:
                    temp = (1 - (1 - h1 ** lamda) ** wa[0] * (1 - h2 ** lamda) ** wa[1] * (1 - h3 ** lamda) ** wa[2] * (
                                1 - h4 ** lamda) ** wa[3]) ** (1 / lamda)
                    GHFWA.append(temp)
    return GHFWA


# 传入a对b的评价


def GHFWA_atob(evalution_list):
    lamda = 1
    GHFWA = []
    for h1 in evalution_list[0]:
        for h2 in evalution_list[1]:
            for h3 in evalution_list[2]:
                temp = (1 - (1 - h1 ** lamda) ** wb[0] * (1 - h2 ** lamda) ** wb[1] * (1 - h3 ** lamda) ** wb[2]) ** (
                            1 / lamda)
                GHFWA.append(temp)
    return GHFWA


# In[] 计算得分和偏差度和满意度

def satisfy_degree(hf):
    score = np.mean(hf)
    deviate = np.std(hf)
    satis = score / (1 + deviate)
    return satis


# In[] 得到满意度矩阵
evalu_matrix_btoa = []
for i in range(a_len * a_attri_num):
    if i % 4 == 0:
        # print(i)
        btoahf = []
        for j in range(b_len):
            temp = []
            temp.append(a_expend[i][j])
            temp.append(a_expend[i + 1][j])
            temp.append(a_expend[i + 2][j])
            temp.append(a_expend[i + 3][j])
            jichenghf = GHFWA_btoa(temp)  # 犹豫模糊集成
            satis_value = satisfy_degree(jichenghf)  # 计算满意度
            btoahf.append(satis_value)
            # 直接集成
        # print(len(btoahf))
        evalu_matrix_btoa.append(btoahf)  # 满意度矩阵

evalu_matrix_atob = []
for i in range(b_len * b_attri_num):
    if i % 3 == 0:
        # print(i)
        btoahf = []
        for j in range(a_len):
            temp = []
            temp.append(b_expend[i][j])
            temp.append(b_expend[i + 1][j])
            temp.append(b_expend[i + 2][j])
            jichenghf = GHFWA_atob(temp)
            satis_value = satisfy_degree(jichenghf)
            btoahf.append(satis_value)
        # print(len(btoahf))
        evalu_matrix_atob.append(btoahf)
# In[]
import pandas as pd
from pandas import DataFrame

Satis_matrix_atob = DataFrame(evalu_matrix_atob)
df = DataFrame(evalu_matrix_btoa)
Satis_matrix_btoa = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)



# In[]
Satis_list_btoa = []
Satis_list_atob = []

data_array = np.array(Satis_matrix_atob)
# 然后转化为list形式
ab = data_array.tolist()
for li in ab:
    Satis_list_atob = Satis_list_atob + li

data_array = np.array(Satis_matrix_btoa)
# 然后转化为list形式
ab = data_array.tolist()
for li in ab:
    Satis_list_btoa = Satis_list_btoa + li

A = Satis_list_atob
B = Satis_list_btoa
a_list = [Satis_list_atob]
b_list = [Satis_list_btoa]

# In[]


n_rows = a_len
n_columns = b_len
# A = [0.619, 0.620, 0.400, 1.000, 0.750, 0.680, 0.611, 0.754, 0.561, 0.756, 0.480, 1.000,0.314, 0.692, 0.650, 0.496, 0.494, 1.000, 0.415, 0.342, 0.568, 0.568, 0.618, 0.448]
#
# B = [0.445, 0.361, 0.490, 0.726, 0.720, 0.655, 0.716, 0.606, 0.507, 0.558, 0.481, 0.573, 0.481, 0.637, 0.633, 0.556, 0.322, 0.373, 0.525, 0.831, 0.485, 0.629, 0.627, 0.574]

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self, M = 2):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        Dim = a_len*b_len # 初始化Dim（决策变量维数）
        maxormins = [-1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-1] * Dim # 决策变量下界
        ub = [2] * Dim # 决策变量上界
        lbin = [0] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop): # 目标函数
        A = np.array(a_list).T
        B = np.array(b_list).T

        f1 = np.dot(pop.Phen, A)
        f2 = np.dot(pop.Phen, B)
        # global f1
        # global f2
        # f1 = np.zeros((100, 1)) #nind
        # f2 = np.zeros((100, 1)) #nind
        # Vars = pop.Phen # 得到决策变量矩阵
        # x = []
        # for i in range(a_len*b_len):
        #     x.append(Vars[:, [i]])
        # global A
        # global B
        #
        # for i in range(a_len*b_len):
        #     f1 = f1 + x[i]*A[i]
        #     f2 = f1 + x[i]*B[i]
        # print(f1)
        # 利用可行性法则处理约束条件
        # cvhstack = []
        # for i in range(a_len * b_len):
        #     if i % b_len == 0:
        #         temp = np.zeros((100, 1))
        #         for j in range(b_len):
        #             temp = temp + x[i + j]
        #         temp = temp - 1
        #         cvhstack.append(temp)
        # for num in range(a_len):
        #     temp = np.zeros((100, 1))
        #     for j in range(b_len):
        #         temp = temp + x[num + j * a_len]
        #     temp = temp - 1
        #     cvhstack.append(temp)
        # pop.CV = np.hstack(cvhstack)
        CV_list = []
        for i in range(n_columns):
            temp = 0
            for j in range(n_rows * i, n_rows * (i + 1)):
                temp += pop.Phen[:, [j]]
            CV_list.append(temp - 1)

        for i in range(n_rows):
            temp = 0
            for j in range(n_columns):
                temp += pop.Phen[:, [i + j * n_rows]]
            CV_list.append(temp - 1)
        pop.CV = np.hstack(CV_list)
        pop.ObjV = np.hstack([f1, f2]) # 把求得的目标函数值赋值给种群pop的ObjV\


