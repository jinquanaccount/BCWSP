
# 数据生成代码仅作为主函数调用一次，以保证后续试验数据的一致性。
# 数据生成时，所有参数从文件中统一读取，不允许使用常数，以避免疏漏，造成生成数据与预想情况不一致。
# 生成的数据存入文件中，不再修改，方便进行模块化组装和测试，这在远程运行代码时同样非常有用。
# 在文件中要进行详细的标注，方便后续程序调用，不要怕浪费存储空间。
# 在定义类的时候，属性与方法尽量分开，避免在初始化的时候执行计算，而是将计算放入方法中，不要贪图方便，要便于理解。
# 上述操作并不会导致重复计算导致和数据错误。实际上，只有初始化会执行这部分计算，赋值时并不会。
# 使用 if __name__ == '__main__': 是防止重复计算的好方法 （import 时不执行，执行则可能会造成数据篡改错误）

import numpy as np
import random
import pickle

Task_number = [100, 200, 300, 400, 500]
Edge_multiple = [2, 3, 4, 5]
Instructions_boundary = [0, 1800]  # 这里假设最慢的机器上的计算时间在0-30分钟。
CCR = [0.5, 0.67, 1.0, 1.5, 2]
Instance_number = 10


class TempTask(object):

    def __init__(self, task_id, children, parents, runtime, total_input_size, total_output_size, transmission):
        self.task_id = task_id  # 为每个任务分配唯一的ID
        self.runtime = runtime
        self.total_input_size = total_input_size
        self.total_output_size = total_output_size
        self.children = children
        self.parents = parents
        self.transmission = transmission


class DAG(object):
    def __init__(self, task_number, edge_multiple, inner_ccr):
        self.task_number = task_number
        self.edge_multiple = edge_multiple
        self.inner_ccr = inner_ccr
        self.task_list = None

    def generator(self):

        dag = np.zeros((self.task_number, self.task_number), dtype=int)
        task_list = []
        # 生成dag时，首先保证每一个节点都有后继节点，从而构成连通图
        for i in range(0, self.task_number - 1):  # 不可以取值为 task_number - 1
            j = random.randint(i + 1, self.task_number - 1)  # 可以取值为 task_number - 1
            dag[i, j] = 1

        edge_counter = self.task_number - 1  # 记录总共生成了多少条边
        # 保证每个节点都有一个前继节点，从而保证入口和出口的唯一性
        for j in range(1, self.task_number):
            flag = True
            for e in dag[:, j]:
                if e != 0:
                    flag = False
                    break

            if flag:
                i = random.randint(0, j - 1)
                dag[i, j] = 1
                edge_counter = edge_counter + 1

        # 增加一些边, 可以证明, 此时的 edge_counter 一定小于 2*task_number
        edge_left = self.task_number * self.edge_multiple - edge_counter
        for _ in range(0, edge_left):  # 随机分配剩余的边数
            while True:
                i = random.randint(0, self.task_number - 2)
                j = random.randint(i + 1, self.task_number - 1)
                if dag[i, j] == 0:
                    dag[i, j] = 1
                    break
        # 添加计算时间,区间缩放
        Instructions_boundary = [0, 1800]
        for i in range(self.task_number):
            task_id = i
            runtime = np.random.rand() * (Instructions_boundary[1] - Instructions_boundary[0])\
                      + Instructions_boundary[0]
            total_data_transmission = runtime / self.inner_ccr * 10 * pow(2, 20) * 2 * np.random.rand()  # 将时间转化为数据量
            datas = np.random.random(3)
            datas = datas / sum(datas) * total_data_transmission
            total_input_size = datas[0] + datas[1]
            total_output_size = datas[2]
            children = []
            parents = []
            for j in range(self.task_number):
                if dag[j, i] == 1:
                    parents.append(j)
                if dag[i, j] == 1:
                    children.append(j)
            transmission = np.random.random(len(parents))
            transmission = transmission / sum(transmission) * datas[0]

            trans = dict()
            for k in range(len(parents)):
                trans[parents[k]] = transmission[k]
            t = TempTask(task_id, children, parents, runtime, total_input_size, total_output_size, trans)
            task_list.append(t)

        self.task_list = task_list


if __name__ == '__main__':
    # 需要存储的数据集
    DagList = []

    counter = 0
    for t_num in Task_number:
        for edge_multi in Edge_multiple:
            for ccr in CCR:
                for in_num in range(Instance_number):
                    graph = DAG(t_num, edge_multi, ccr)
                    graph.generator()
                    DagList.append(graph)
                    counter = counter + 1
                    print(counter)
                    path = 'G:\TrainBC\\' + str(t_num) + '_' + str(edge_multi) + '_' + str(ccr) + '_' + str(in_num)
                    with open(path, 'wb') as dbfile:
                        pickle.dump(graph, dbfile)

