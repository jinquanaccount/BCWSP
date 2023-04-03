from Resources.job import Job
from Resources.cluster import Cluster
from Resources.scheduler_comparison20230216 import MultiScheduler
from Resources.machine import Machine
from JsonDataGenerationBC import DAG, TempTask
from multiprocessing import Pool, cpu_count, Value, Manager
import numpy as np
import pickle
import os
import csv
import copy
import time
from tqdm import tqdm
import random

random.seed(10)

Remote = True
if Remote:
    DataSetPath = "/home/jqzhang/Datasets/TestBC"  # 设置数据集路径
else:
    DataSetPath = "G:\TestBC"    # 设置数据集路径
files = os.listdir(DataSetPath)    # 读取文件夹下面的所有文件的名称
files.sort()
files = files[0:200] + files[250:450] + files[500:700] + files[750:950] \
        + files[1000:1200] + files[1250:1450] + files[1500:1700] + files[1750:1950]  # 仅对100-400节点数目进行测试
# files = random.sample(files, 200)  # 随机切分数据，验证不同算法的效果
ResultFile = "PerformanceTest20230216.csv"  # 定义结果存储文件
Machine_Types = [x for x in range(len(Machine.prices))]
BudgetFactorList = [0.1, 0.2, 0.3, 0.4, 0.5]  # 在最大值和最小值之间浮动，分别通过最贵的和最便宜的函数计算。
header = ['Recipe', 'TaskNumber', 'CCR', 'InstanceNumber', 'Machine_Type', 'BudgetFactor', 'Algorithm', 'Makespan',
          'Time']
AlgorithmList = ['CG', 'SMOHEFT', 'GRP-HEFT', 'BCWS', 'GRP-HEFT_HE']
algo_number = len(AlgorithmList)


def load(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        return pickle_data


def WriteData(file):
    workflow = load(DataSetPath + "/" + file)
    InformationList = file.split("_")  # 这里对随机生成的文件有名称要求
    Recipe = InformationList[0]
    TaskNumber = InformationList[1]
    CCR = InformationList[2]
    InstanceNumber = InformationList[3]
    job = Job(workflow, "train")
    for bf in BudgetFactorList:  # 工作流计算量和通信量的分析应该放在这部分代码之前，时间参数的计算要放在这部分代码之后
        BudgetFactor = bf
        for machine_type in Machine_Types:
            Machine.price = Machine.prices[machine_type]
            Machine.speed = Machine.speeds[machine_type]
            for alg in AlgorithmList:
                Algorithm = alg
                inner_job = copy.deepcopy(job)
                t1 = time.time()
                inner_job.SetBudgetFactor(bf, Machine.price, Machine.speed)  # 为任务设定预算。
                cluster = Cluster()
                scheduler = MultiScheduler(inner_job, cluster)  # 先定义各个类的成员，然后再执行方法。
                Makespan = scheduler.run(Algorithm)
                t2 = time.time()
                TimeConsuming = t2 - t1
                my_list.append([Recipe, TaskNumber, CCR, InstanceNumber, machine_type, BudgetFactor, Algorithm,
                                Makespan, TimeConsuming])
    v.value = v.value + 1
    print('子进程: {}，'.format(os.getpid()) + '第{}个实例'.format(v.value) + ' 已完成！')


v = Value("d", 0)
if Remote:
    manager = Manager()
    my_list = manager.list()  # 用列表及其排序控制文件写入顺序
    p = Pool(60)  # 这里确定进程池中的进程数目
    for FileName in files:
        p.apply_async(WriteData, args=(FileName,))
    print('等待所有子进程完成。')
    p.close()
    p.join()
else:
    my_list = []
    for FileName in files:
        WriteData(FileName)

my_list = sorted(my_list, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]))
csvfile = open(ResultFile, 'w+', newline='')  # 定义结果存储文件, 必须在子进程中打开和关闭，否则后续进程会无法写入。
writer = csv.writer(csvfile, delimiter=",")
writer.writerow(header)
for ele in my_list:
    writer.writerow(ele)
csvfile.close()
