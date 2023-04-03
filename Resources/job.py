from task import Task
import numpy as np
import math
from faas import Faas
from machine import Machine


class Job:
    def __init__(self, wf_instance, mode):
        if mode == "test":
            pass
        if mode == "train":
            self.task_number = wf_instance.task_number
            self.task_list = []
            for t in wf_instance.task_list:
                task_id = t.task_id
                children = t.children.copy()
                parents = t.parents.copy()
                instructions = t.runtime*np.mean(Faas.speeds)
                total_input_size = t.total_input_size
                total_output_size = t.total_output_size
                task = Task(self, task_id, children, parents, instructions, total_input_size, total_output_size)
                task.transmission = t.transmission.copy()
                self.task_list.append(task)
            self.BudgetFactor = None
            self.Budget = None
            self.queue = None
            self.Task_to_Function = None
            self.Task_to_VM = None
            self.parallel = None

    def SetBudgetFactor(self, BudgetFactor, price, speed):
        Machine.price = price
        Machine.speed = speed
        self.BudgetFactor = BudgetFactor
        '''min_budget, max_budget = 0, 0
        for t in self.task_list:
            min_budget = min_budget + t.instructions / Faas.speeds[0] + t.max_data_time
            max_budget = max_budget + t.instructions / Faas.speeds[-1] + t.max_data_time
        min_budget, max_budget = min_budget/Faas.unit*Faas.prices[0], max_budget/Faas.unit*Faas.prices[-1]
        self.Budget = min_budget + self.BudgetFactor*(max_budget-min_budget)'''
        min_budget = 0
        for t in self.task_list:
            min_budget = min_budget + t.instructions / Faas.speeds[0] + t.max_data_time
        min_budget = min_budget / Faas.unit * Faas.prices[0]

        trans_data = 0
        max_trans_time = 0
        total_runtime_time = 0
        for t in self.task_list:
            max_trans_time = max_trans_time + t.max_data_time
            total_runtime_time = total_runtime_time + t.instructions / Machine.speed
            for trans in t.transmission:
                trans_data = trans_data + trans
        total_execution_time = total_runtime_time + max_trans_time - trans_data/Task.NetworkBandwidth
        min_budget_VM = np.ceil(total_execution_time/Machine.unit)*Machine.price
        min_budget = max(min_budget, min_budget_VM)

        self.Budget = min_budget * (1 + self.BudgetFactor)

    def ParallelDegree(self):
        f = np.zeros([self.task_number, self.task_number])
        for i in range(self.task_number):  # 构造邻接矩阵
            task = self.task_list[i]
            for c in task.children:
                f[task.task_id][c] = 1
        for k in range(self.task_number):  # floyd算法计算有向图的传递闭包
            for s in range(self.task_number):
                for x in range(self.task_number):
                    if f[s][k] and f[k][x]:
                        f[s][x] = 1
        parallel = np.zeros(self.task_number)
        for i in range(self.task_number):
            parallel[i] = self.task_number - (sum(f[i, :]) + sum(f[:, i]))
        parallel = parallel / self.task_number  # 可并行任务数归一化
        self.parallel = parallel
