from Resources.job import Job
from Resources.cluster import Cluster
from Resources.scheduler20230216 import Scheduler
import pickle
from Resources.machine import Machine
from Resources.RandomData import DAG, TempTask

from multiprocessing import Pool, cpu_count, Value, Manager
import os
import csv
import copy
import time
from tqdm import tqdm
import random

random.seed(10)

Remote = True
if Remote:
    DataSetPath = "/home/jqzhang/Datasets/TrainBC"  # 设置数据集路径
else:
    DataSetPath = "G:\TrainBC"
files = os.listdir(DataSetPath)  # 读取文件夹下面的所有文件的名称
files.sort()
files = files[0:800]
filtered_files = []
for k in range(len(files)):
    if k % 10 < 3:
        filtered_files.append(files[k])
files = filtered_files
# files = random.sample(files, 60)  # 不需要随机选取。
Machine_Types = [x for x in range(len(Machine.prices))]
BudgetFactorList = [0.1, 0.2, 0.3, 0.4, 0.5]
Components = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
ISC = ['MCETRF']
Queue = ['ESTF']  # UB的情况下最好还是减少迭代轮数，不然时间太长。
BTU = ['FRN4']
BR = ['GTA']
header = ['TaskNumber', 'LinkDensity', 'CCR', 'InstanceNumber', 'Machine_Type', 'BudgetFactor', 'Components',
          'ISC', 'Queue', 'BTU', 'BR', 'Makespan', 'Time']


def load(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        return pickle_data

if Remote:
    PDPath = "/home/jqzhang/Datasets/TrainPD"
else:
    PDPath = "G:\TrainPD"
PD_files = os.listdir(PDPath)
ResultFile = "Effectiveness_20230216.csv"


def WriteData(file):
    workflow = load(DataSetPath + "/" + file)
    InformationList = file.split("_")  # 这里对随机生成的文件有名称要求
    TaskNumber = float(InformationList[0])
    LinkDensity = float(InformationList[1])
    CCR = float(InformationList[2])
    InstanceNumber = float(InformationList[3])
    job = Job(workflow, "train")
    if file in PD_files:
        job.parallel = load(PDPath + "/" + file)
    else:
        job.ParallelDegree()
        path = PDPath + '/' + file
        with open(path, 'wb') as dbfile:
            pickle.dump(job.parallel, dbfile)
            return
    temp_list = []
    for machine_type in Machine_Types:
        Machine.price = Machine.prices[machine_type]
        Machine.speed = Machine.speeds[machine_type]
        for bf in BudgetFactorList:  # 工作流计算量和通信量的分析应该放在这部分代码之前，时间参数的计算要放在这部分代码之后
            BudgetFactor = bf
            for com in Components:
                for isc in ISC:
                    for q in Queue:
                        for b in BTU:
                            for br in BR:
                                G1 = com[0]
                                G2 = com[1]
                                G3 = com[2]  # 不同的组合
                                Comp = 4 * G1 + 2 * G2 + G3
                                inner_job = copy.deepcopy(job)
                                t1 = time.time()
                                inner_job.SetBudgetFactor(bf, Machine.price, Machine.speed)  # 为任务设定预算。
                                # inner_job.ParallelDegree() # 这部分代码放到最前面就可以了
                                cluster = Cluster()
                                scheduler = Scheduler(inner_job, cluster, G1, G2, G3, isc, q, b, br, Machine.price, Machine.speed)
                                Makespan = scheduler.run()
                                t2 = time.time()
                                TimeConsuming = t2 - t1
                                my_list.append([TaskNumber, LinkDensity, CCR, InstanceNumber, machine_type, BudgetFactor, Comp,
                                                 isc, q, b, br, Makespan, TimeConsuming])
                                temp_list.append([TaskNumber, LinkDensity, CCR, InstanceNumber, machine_type, BudgetFactor, Comp,
                                                 isc, q, b, br, Makespan, TimeConsuming, file])
    v.value = v.value + 1
    print('子进程: {}，'.format(os.getpid()) + '第{}个实例'.format(v.value) + ' 已完成！')


v = Value("d", 0)
if Remote:
    manager = Manager()
    my_list = manager.list()  # 用列表及其排序控制文件写入顺序
else:
    my_list = []

print('CPU内核数: {}'.format(cpu_count()))
print('当前母进程: {}'.format(os.getpid()))
start = time.time()

if Remote:
    p = Pool(60)  # 这里确定进程池中的进程数目
    for FileName in files:
        p.apply_async(WriteData, args=(FileName,))
    print('等待所有子进程完成。')
    p.close()
    p.join()
else:
    for FileName in files:
        WriteData(FileName)

end = time.time()
print("总共用时{}秒".format((end - start)))
my_list = sorted(my_list, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]))
csvfile = open(ResultFile, 'w+', newline='')  # 定义结果存储文件, 必须在子进程中打开和关闭，否则后续进程会无法写入。
writer = csv.writer(csvfile, delimiter=",")
writer.writerow(header)
for ele in my_list:
    writer.writerow(ele)
csvfile.close()
print("Writing Process Finished!")
