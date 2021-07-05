import geatpy as ea # import geatpy
from MyProblem import MyProblem # 导入自定义问题接口
import random
# random.seed(128)
if __name__ == '__main__':
    """================================实例化问题对象==========================="""
    problem = MyProblem()       # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'BG'             # 编码方式
    NIND = 50                   # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA3_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.1 # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.8# 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 500   # 最大进化代数
    myAlgorithm.logTras = 50
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 1
    """==========================调用算法模板进行种群进化========================"""
    [NDSet, population] = myAlgorithm.run()    # 执行算法模板，得到帕累托最优解集NDSet
    NDSet.save()                # 把结果保存到文件中
    # 输出
    print('total running time：%s second'%(myAlgorithm.passTime))
    # print('非支配个体数：%s 个'%(NDSet.sizes))
    # if NDSet.sizes != 0:
    #     print('最优的目标函数值为：%s' % NDSet.ObjV[0][0])
    #     # print('最优的控制变量值为：')
    #     # for i in range(NDSet.Phen.shape[1]):
    #     #     print(NDSet.Phen[0, i])
    # else:
    #     print('没找到可行解。')
