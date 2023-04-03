import copy
from machine import Machine
from faas import Faas
from task import Task
import numpy as np
from plotGantt import gantt


class MultiScheduler:
    def __init__(self, job, cluster):
        self.job_bak = job
        self.cluster_bak = cluster
        self.job = None
        self.cluster = None
        self.money = 0
        self.makespan = 0
        self.isc = "MCETRF"
        self.q = 'ESTF'
        self.b = 'FRN4'
        self.br = 'GTA'
        self.G1 = True
        self.G2 = True
        self.G3 = True

    def run(self, Algorithm):
        self.job = copy.deepcopy(self.job_bak)
        self.cluster = copy.deepcopy(self.cluster_bak)
        if Algorithm == "CG":
            RemainingCost, Delay = self.CG()
            self.makespan = Delay
            return self.makespan
        if Algorithm == "SMOHEFT":
            self.makespan = self.SMOHEFT()
            return self.makespan
        if Algorithm == "GRP-HEFT":
            self.makespan = self.GRPHEFT()
            return self.makespan
        if Algorithm == "BCWS":
            RemainingCost, self.makespan = self.BCWS()
            return self.makespan
        if Algorithm == "GRP-HEFT_HE":
            self.makespan = self.GRPHEFT_HE()
            return self.makespan

    def ready_time(self, task):
        ready = 0
        for p in task.parents:
            parent = self.job.task_list[p]
            if ready < parent.end_time:
                ready = parent.end_time
        return ready

    def CG(self):
        def CalculateGBL():
            cost_min, cost_max = 0, 0
            for task in self.job.task_list:
                vt_min = (task.instructions / Faas.speeds[0] + task.max_data_time) * Faas.prices[0] / Faas.unit
                vt_max = (task.instructions / Faas.speeds[-1] + task.max_data_time) * Faas.prices[-1] / Faas.unit
                cost_min = cost_min + vt_min
                cost_max = cost_max + vt_max
            return (self.job.Budget-cost_min)/(cost_max-cost_min)

        def selectInitType(task, global_budget_level):
            vt_min = (task.instructions / Faas.speeds[0] + task.max_data_time) * Faas.prices[0] / Faas.unit
            vt_max = (task.instructions / Faas.speeds[-1] + task.max_data_time) * Faas.prices[-1] / Faas.unit
            target_cost = vt_min + (vt_max - vt_min) * global_budget_level
            selected_fun_type = 0
            for fun_type in range(len(Faas.speeds)):
                if (task.instructions / Faas.speeds[fun_type] + task.max_data_time) * \
                        Faas.prices[fun_type] / Faas.unit <= target_cost:
                    selected_fun_type = fun_type
                else:
                    break
            return selected_fun_type

        def CriticalPath(total_delay):
            EST, LST = np.zeros(self.job.task_number), np.zeros(self.job.task_number)
            for i in range(self.job.task_number):
                task = self.job.task_list[i]
                EST[i] = task.start_time
            for i in range(self.job.task_number):
                task = self.job.task_list[self.job.task_number - i - 1]
                if not task.children:
                    LST[task.task_id] = total_delay - task.execution_time
                else:
                    latest_finish_time = float("inf")
                    for c in task.children:
                        child_lst = LST[c]
                        if latest_finish_time > child_lst:
                            latest_finish_time = child_lst
                    LST[task.task_id] = latest_finish_time - task.execution_time
            CriticalTasks = []
            for i in range(self.job.task_number):
                if abs(LST[i] - EST[i]) < 1e-6:
                    CriticalTasks.append(i)
            return CriticalTasks

        def Critical_Allocate(CostToSpend, total_delay):
            while CostToSpend > 0:
                CriticalTasks = CriticalPath(total_delay)
                max_indicate, max_t_id, max_ftype = 0, 0, 0
                for t_id in CriticalTasks:
                    task_inner = self.job.task_list[t_id]
                    time_current = task_inner.execution_time
                    cost_current_inner = time_current / Faas.unit * Faas.prices[selected_type_list[task_inner.task_id]]
                    for ftype in range(selected_type_list[t_id] + 1, len(Faas.speeds)):
                        time_new = task_inner.instructions / Faas.speeds[ftype] + task_inner.max_data_time
                        cost_new = time_new / Faas.unit * Faas.prices[ftype]
                        delta_time = time_current - time_new
                        delta_cost = cost_new - cost_current_inner
                        indicate = delta_time / delta_cost
                        if max_indicate < indicate and delta_cost < CostToSpend:
                            max_indicate, max_t_id, max_ftype = indicate, t_id, ftype
                if max_indicate == 0:
                    break
                # Map max_t_id to max_ftype
                task = self.job.task_list[max_t_id]
                time_current = task.execution_time
                cost_current_inner = time_current / Faas.unit * Faas.prices[selected_type_list[max_t_id]]
                time_new = task.instructions / Faas.speeds[max_ftype] + task.max_data_time
                cost_new = time_new / Faas.unit * Faas.prices[max_ftype]
                task.execution_time = time_new
                CostToSpend = CostToSpend - (cost_new - cost_current_inner)
                selected_type_list[max_t_id] = max_ftype
                total_delay = 0
                for task_inner in self.job.task_list:
                    task_inner.start_time = self.ready_time(task_inner)
                    task_inner.end_time = task_inner.start_time + task_inner.execution_time
                    if total_delay < task_inner.end_time:
                        total_delay = task_inner.end_time
            return CostToSpend, total_delay

        gbl = CalculateGBL()  # CG算法中，计算global budget level 的计算结果与BudgetFactor是相同的
        selected_type_list = []
        for t in self.job.task_list:
            selected_type = selectInitType(t, gbl)
            selected_type_list.append(selected_type)
            t.execution_time = t.instructions / Faas.speeds[selected_type] + t.max_data_time
            t.start_time = self.ready_time(t)
            t.end_time = t.start_time + t.execution_time
        # 计算end-to-end delay 和 cost_current
        end2end_delay, cost_current = 0, 0
        for t in self.job.task_list:
            if end2end_delay < t.end_time:
                end2end_delay = t.end_time
            cost_current = cost_current + t.execution_time / Faas.unit * Faas.prices[selected_type_list[t.task_id]]
        return Critical_Allocate(self.job.Budget - cost_current, end2end_delay)

    def SMOHEFT(self):
        class Solution:
            def __init__(self, task_number):
                self.current_task = 0
                self.cost = 0
                self.execution_time = 0
                self.total_cost = 0
                self.makespan = 0
                self.remaining_budget = float("inf")
                self.FT = np.zeros(task_number)
                self.distance = 0

        def CalculateGBL():
            cost_min, cost_max = 0, 0
            for task in self.job.task_list:
                vt_min = (task.instructions / Faas.speeds[0] + task.max_data_time) * Faas.prices[0] / Faas.unit
                vt_max = (task.instructions / Faas.speeds[-1] + task.max_data_time) * Faas.prices[-1] / Faas.unit
                cost_min = cost_min + vt_min
                cost_max = cost_max + vt_max
            return (self.job.Budget-cost_min)/(cost_max-cost_min)

        def selectInitType(task, global_budget_level):
            vt_min = (task.instructions / Faas.speeds[0] + task.max_data_time) * Faas.prices[0] / Faas.unit
            vt_max = (task.instructions / Faas.speeds[-1] + task.max_data_time) * Faas.prices[-1] / Faas.unit
            target_cost = vt_min + (vt_max - vt_min) * global_budget_level
            selected_fun_type = 0
            for fun_type in range(len(Faas.speeds)):
                if (task.instructions / Faas.speeds[fun_type] + task.max_data_time) * \
                        Faas.prices[fun_type] / Faas.unit < target_cost:
                    selected_fun_type = fun_type
                else:
                    break
            return selected_fun_type

        def TaskSequencing():
            AET = []
            for t in self.job.task_list:
                total_execution_time = 0
                for speed in Faas.speeds:
                    total_execution_time = total_execution_time + t.instructions / speed + t.max_data_time
                average_execution_time = total_execution_time / len(Faas.speeds)
                AET.append(average_execution_time)
            rank = np.zeros(self.job.task_number)
            for i in range(self.job.task_number):
                current_task = self.job.task_list[self.job.task_number - i - 1]
                for child in current_task.children:
                    p = self.job.task_list[child].priority
                    if p > current_task.priority:
                        current_task.priority = p
                current_task.priority = current_task.priority + AET[current_task.task_id]
                rank[self.job.task_number - i - 1] = current_task.priority
            self.job.queue = np.argsort(-np.array(rank))

        def LeastRemainingBudget():
            LRBList = []
            sum_cost = 0
            LRBList.append(sum_cost)
            for i in range(self.job.task_number):
                t_id = self.job.queue[self.job.task_number - i - 1]
                task = self.job.task_list[t_id]
                base_type = selected_type_list[t_id]
                sum_cost = sum_cost + (task.instructions / Faas.speeds[base_type] + task.max_data_time) / Faas.unit * \
                           Faas.prices[base_type]
                LRBList.append(sum_cost)
            LRBList.pop()
            LRBList.reverse()
            return LRBList

        def CrowdingDistance(S_temp):
            # 筛选非支配解
            S_temp_count = np.zeros(len(S_temp))
            S_temp_temp = []
            for i in range(len(S_temp)):
                for j in range(len(S_temp)):
                    if (S_temp[i].makespan < S_temp[j].makespan) and (S_temp[i].total_cost < S_temp[j].total_cost):
                        S_temp_count[j] += 1
            for i in range(len(S_temp)):
                if S_temp_count[i] == 0:
                    S_temp_temp.append(S_temp[i])
            S_temp = S_temp_temp
            # 拥挤距离排序前的准备工作
            makespans, total_costs = [], []
            solution_number = len(S_temp)
            for i in range(solution_number):
                makespans.append(S_temp[i].makespan)
                total_costs.append(S_temp[i].total_cost)
            min_makespan, max_makespan = min(makespans), max(makespans)
            min_cost, max_cost = min(total_costs), max(total_costs)
            # 按照makespan进行排序
            S_temp.sort(key=lambda x: x.makespan)
            S_temp[0].distance = float("inf")
            S_temp[solution_number - 1].distance = float("inf")
            for i in range(1, solution_number - 1):
                S_temp[i].distance = S_temp[i].distance + (S_temp[i + 1].makespan - S_temp[i - 1].makespan) / \
                                     (max_makespan - min_makespan + 1)
            # 按照cost进行排序

            S_temp.sort(key=lambda x: x.total_cost)
            S_temp[0].distance = float("inf")
            S_temp[solution_number - 1].distance = float("inf")
            for i in range(1, solution_number - 1):
                S_temp[i].distance = S_temp[i].distance + (S_temp[i + 1].total_cost - S_temp[i - 1].total_cost) / \
                                     (max_cost - min_cost + 1)
            return S_temp

        def SolutionSet(K, Budget, LRB):
            S_list = []
            for i in range(self.job.task_number):
                t_id = self.job.queue[i]
                task = self.job.task_list[t_id]
                S_temp_list = []
                base_type = selected_type_list[t_id]
                if not S_list:
                    for j in range(len(Faas.speeds[base_type:])):
                        execution_time = task.instructions / Faas.speeds[j+base_type] + task.max_data_time
                        cost = execution_time / Faas.unit * Faas.prices[j+base_type]
                        if Budget - cost >= LRB[0]:
                            new_solution = Solution(self.job.task_number)
                            new_solution.current_task = 0
                            new_solution.execution_time = execution_time
                            new_solution.cost = cost
                            new_solution.makespan = execution_time
                            new_solution.total_cost = cost
                            new_solution.remaining_budget = Budget - new_solution.cost
                            new_solution.FT[t_id] = new_solution.execution_time
                            S_temp_list.append(new_solution)
                        else:
                            break
                else:
                    for s in S_list:
                        for j in range(len(Faas.speeds[base_type:])):
                            execution_time = task.instructions / Faas.speeds[j+base_type] + task.max_data_time
                            cost = execution_time / Faas.unit * Faas.prices[j+base_type]
                            if s.remaining_budget - cost >= LRB[i]:
                                new_solution = copy.deepcopy(s)
                                new_solution.current_task = t_id
                                new_solution.execution_time = execution_time
                                new_solution.cost = cost
                                start_time = 0
                                for p in task.parents:
                                    if start_time < new_solution.FT[p]:
                                        start_time = new_solution.FT[p]
                                new_solution.makespan = max(new_solution.makespan, start_time + execution_time)
                                new_solution.total_cost = new_solution.total_cost + cost
                                new_solution.remaining_budget = new_solution.remaining_budget - new_solution.cost
                                new_solution.FT[t_id] = start_time + new_solution.execution_time
                                S_temp_list.append(new_solution)
                S_temp_list = CrowdingDistance(S_temp_list)
                S_temp_list.sort(key=lambda x: x.distance)
                S_list = copy.deepcopy(S_temp_list[0:K])  # 此处没有问题，如果列表长度不够会自动截断
            S_list.sort(key=lambda x: x.makespan)
            return S_list[0]

        gbl = CalculateGBL()
        selected_type_list = []
        for t in self.job.task_list:
            selected_type = selectInitType(t, gbl)
            selected_type_list.append(selected_type)
        TaskSequencing()
        LRB = LeastRemainingBudget()  # 产生可行解的保证。
        BestSolution = SolutionSet(10, self.job.Budget, LRB)  # 10是由MOHEFT给出的超参数。
        return BestSolution.makespan

    def GRPHEFT(self):  # 注意：改算法可能会违反预算约束，且经常出现，可以计算超出部分的比例。
        def TaskSequencing():
            AET = []
            for t in self.job.task_list:
                average_execution_time = t.instructions / Machine.speed + t.max_data_time
                AET.append(average_execution_time)
            rank = np.zeros(self.job.task_number)
            for i in range(self.job.task_number):
                current_task = self.job.task_list[self.job.task_number - i - 1]
                for child in current_task.children:
                    p = self.job.task_list[child].priority
                    if p > current_task.priority:
                        current_task.priority = p
                current_task.priority = current_task.priority + AET[current_task.task_id]
                rank[self.job.task_number - i - 1] = current_task.priority
            self.job.queue = np.argsort(-np.array(rank))

        def SavedTime(instance_inner, task_inner):
            parents, queue, save = set(task_inner.parents), set(), 0
            for t in instance_inner.queue:
                queue.add(t.task_id)
            parents_local = list(parents.intersection(queue))
            for p in parents_local:
                save = save + task_inner.transmission[p] / task_inner.NetworkBandwidth
            return save

        units = np.floor(self.job.Budget / Machine.price)  # 计算可租赁的周期数
        for i in range(int(units)):  # 构造资源集合
            machine = Machine()
            self.cluster.machine_unused.append(machine)
        TaskSequencing()
        for i in range(self.job.task_number):
            t_id = self.job.queue[i]
            task = self.job.task_list[t_id]
            I_min, I_min_star, selected_VM = None, None, None
            FT_min, FT_min_star = float("inf"), float("inf")
            I_min_cost_increment = 0
            for instance in self.cluster.machine_unused:
                start_time = self.ready_time(task)
                finish_time = start_time + task.instructions / Machine.speed + task.max_data_time
                if finish_time < FT_min:
                    I_min = instance
                    FT_min = finish_time
                    I_min_cost_increment = np.ceil(finish_time / Machine.unit) * instance.price
            for instance in self.cluster.machine_running:
                start_time = max(instance.avail_time(), self.ready_time(task))
                execution_time = task.instructions / Machine.speed + task.max_data_time - SavedTime(instance, task)
                finish_time = start_time + execution_time
                current_cost = instance.cost
                new_cost = np.ceil((finish_time - instance.start_time) / Machine.unit) * instance.price
                if finish_time < FT_min:
                    I_min = instance
                    FT_min = finish_time
                    I_min_cost_increment = new_cost - current_cost
                if (current_cost == new_cost) and finish_time < FT_min_star:
                    I_min_star = instance
                    FT_min_star = finish_time
            if I_min is not I_min_star:
                remaining_units = len(self.cluster.machine_unused)
                remaining_budget = remaining_units * Machine.price
                if I_min_cost_increment <= remaining_budget:
                    external_units = round(I_min_cost_increment / Machine.price)
                    self.cluster.machine_unused = self.cluster.machine_unused[0:int(remaining_units - external_units)]
                else:
                    if I_min_star is not None:
                        task.start_time = max(I_min_star.avail_time(), self.ready_time(task))
                        task.execution_time = task.instructions / Machine.speed + task.max_data_time - SavedTime(
                            I_min_star, task)
                        task.end_time = task.start_time + task.execution_time
                        task.scheduled = True
                        I_min_star.add_task(task)

            if not task.scheduled:
                if I_min not in self.cluster.machine_running:
                    I_min = Machine()
                    self.cluster.machine_running.append(I_min)
                # 这里的I_min可能是一个新的实例或者已有的实例
                task.start_time = max(I_min.avail_time(), self.ready_time(task))
                task.execution_time = task.instructions / Machine.speed + task.max_data_time - SavedTime(I_min, task)
                task.end_time = task.start_time + task.execution_time
                task.scheduled = True
                I_min.add_task(task)
        makespan = 0
        for i in range(self.job.task_number):
            finish_time = self.job.task_list[i].end_time
            if makespan < finish_time:
                makespan = finish_time
        return makespan

    def GRPHEFT_HE(self):
        def TaskSequencing():
            AET = []
            for t in self.job.task_list:
                average_execution_time = t.instructions / np.average(Machine.speeds) + t.max_data_time
                AET.append(average_execution_time)
            rank = np.zeros(self.job.task_number)
            for i in range(self.job.task_number):
                current_task = self.job.task_list[self.job.task_number - i - 1]
                for child in current_task.children:
                    p = self.job.task_list[child].priority
                    if p > current_task.priority:
                        current_task.priority = p
                current_task.priority = current_task.priority + AET[current_task.task_id]
                rank[self.job.task_number - i - 1] = current_task.priority
            self.job.queue = np.argsort(-np.array(rank))

        def SavedTime(instance_inner, task_inner):
            parents, queue, save = set(task_inner.parents), set(), 0
            for t in instance_inner.queue:
                queue.add(t.task_id)
            parents_local = list(parents.intersection(queue))
            for p in parents_local:
                save = save + task_inner.transmission[p] / task_inner.NetworkBandwidth
            return save

        def TestMakespan(tp):
            fun_number = len(Machine.speeds)
            Rented_units = [0 for _ in range(fun_number)]
            RB = self.job.Budget
            while tp >= 0:  # 计算每种类型的VM租赁多少个BTU
                current_speed = Machine.speeds[tp]
                current_price = Machine.prices[tp]
                current_units = np.floor(RB / current_price)
                Rented_units[tp] = int(round(current_units))
                RB = RB - current_units * current_price
                tp = tp - 1
            for k in range(fun_number):
                for _ in range(Rented_units[k]):
                    machine = Machine()
                    machine.speed = Machine.speeds[k]
                    machine.price = Machine.speeds[k]
                    self.cluster.machine_unused.append(machine)
            TaskSequencing()
            for i in range(self.job.task_number):
                t_id = self.job.queue[i]
                task = self.job.task_list[t_id]
                I_min, I_min_star, selected_VM = None, None, None
                FT_min, FT_min_star = float("inf"), float("inf")
                I_min_cost_increment = 0
                for instance in self.cluster.machine_unused:
                    start_time = self.ready_time(task)
                    finish_time = start_time + task.instructions / instance.speed + task.max_data_time
                    if finish_time < FT_min:
                        I_min = instance
                        FT_min = finish_time
                        I_min_cost_increment = np.ceil(finish_time / Machine.unit) * instance.price
                for instance in self.cluster.machine_running:
                    start_time = max(instance.avail_time(), self.ready_time(task))
                    execution_time = task.instructions / instance.speed + task.max_data_time - SavedTime(instance, task)
                    finish_time = start_time + execution_time
                    current_cost = instance.cost
                    new_cost = np.ceil((finish_time - instance.start_time) / Machine.unit) * instance.price
                    if finish_time < FT_min:
                        I_min = instance
                        FT_min = finish_time
                        I_min_cost_increment = new_cost - current_cost
                    if (current_cost == new_cost) and finish_time < FT_min_star:
                        I_min_star = instance
                        FT_min_star = finish_time
                flag = True
                if I_min is not I_min_star:
                    ty = Machine.speeds.index(I_min.speed)
                    remaining_units = Rented_units[ty]
                    if I_min_cost_increment <= remaining_units*I_min.price:
                        flag = False
                        external_units = int(round(I_min_cost_increment / I_min.price))
                        remove_list = []
                        count = 0
                        for ele in self.cluster.machine_unused:
                            if count < external_units\
                                    and ele.speed == I_min.speed:
                                remove_list.append(ele)
                                count = count + 1
                        for ele in remove_list:
                            self.cluster.machine_unused.remove(ele)
                        Rented_units[ty] = Rented_units[ty] - external_units
                    else:
                        if I_min_star is not None:
                            task.start_time = max(I_min_star.avail_time(), self.ready_time(task))
                            task.execution_time = task.instructions / I_min_star.speed + task.max_data_time - SavedTime(
                                I_min_star, task)
                            task.end_time = task.start_time + task.execution_time
                            task.scheduled = True
                            I_min_star.add_task(task)

                if not task.scheduled:
                    if flag:
                        ty = Machine.speeds.index(I_min.speed)
                        Rented_units[ty] = 0
                        remove_list = []
                        for ele in self.cluster.machine_unused:
                            if ele.speed == I_min.speed:
                                remove_list.append(ele)
                        for ele in remove_list:
                            self.cluster.machine_unused.remove(ele)
                    if I_min not in self.cluster.machine_running:
                        speed = I_min.speed
                        price = I_min.price
                        I_min = Machine()
                        I_min.speed = speed
                        I_min.price = price
                        self.cluster.machine_running.append(I_min)
                    # 这里的I_min可能是一个新的实例或者已有的实例
                    task.start_time = max(I_min.avail_time(), self.ready_time(task))
                    task.execution_time = task.instructions / I_min.speed + task.max_data_time - SavedTime(I_min, task)
                    task.end_time = task.start_time + task.execution_time
                    task.scheduled = True
                    I_min.add_task(task)
            makespan = 0
            for i in range(self.job.task_number):
                finish_time = self.job.task_list[i].end_time
                if makespan < finish_time:
                    makespan = finish_time
            return makespan

        m_min = float("inf")
        for t in range(len(Machine.speeds)):
            self.job_bak = self.job
            self.cluster_bak = self.cluster
            self.job = copy.deepcopy(self.job_bak)
            self.cluster = copy.deepcopy(self.cluster_bak)
            m = TestMakespan(len(Machine.speeds) - t - 1)  # 性价比排名已经确定
            if m < m_min:
                m_min = m
        return m_min

    def BCWS(self):

        def CriticalPath(total_delay):
            EST, LST = np.zeros(self.job.task_number), np.zeros(self.job.task_number)
            for i in range(self.job.task_number):
                task = self.job.task_list[i]
                EST[i] = task.start_time
            for i in range(self.job.task_number):
                task = self.job.task_list[self.job.task_number - i - 1]
                if not task.children:
                    LST[task.task_id] = total_delay - task.execution_time
                else:
                    latest_finish_time = float("inf")
                    for c in task.children:
                        child_lst = LST[c]
                        if latest_finish_time > child_lst:
                            latest_finish_time = child_lst
                    LST[task.task_id] = latest_finish_time - task.execution_time
            CriticalTasks, Task_CCR = [], []
            for i in range(self.job.task_number):
                if abs(LST[i] - EST[i]) < 1e-6:
                    CriticalTasks.append(i)
                et = self.job.task_list[i].execution_time
                data_time = self.job.task_list[i].max_data_time
                Task_CCR.append(abs(et - data_time) / et)
            return CriticalTasks, Task_CCR

        def Critical_Allocate(CostToSpend, total_delay):
            while CostToSpend > 0:
                CriticalTasks, Task_CCR = CriticalPath(total_delay)
                max_indicate, max_t_id, max_ftype = 0, 0, 0
                for t_id in CriticalTasks:
                    task_inner = self.job.task_list[t_id]
                    time_current = task_inner.execution_time
                    cost_current_inner = time_current / Faas.unit * Faas.prices[selected_type_list[task_inner.task_id]]
                    # 当CPU资源与价格呈现线性关系时，不需要对类型进行搜索，选择最近的类型即可。
                    if selected_type_list[t_id] + 1 < len(Faas.speeds):
                        ftype = selected_type_list[t_id] + 1
                        time_new = task_inner.instructions / Faas.speeds[ftype] + task_inner.max_data_time
                        cost_new = time_new / Faas.unit * Faas.prices[ftype]

                        delta_time = time_current - time_new
                        delta_cost = cost_new - cost_current_inner

                        if self.isc == 'MTDCIRF':
                            indicate = delta_time / delta_cost
                        elif self.isc == 'MCETRF':
                            indicate = Task_CCR[t_id]
                        else:
                            # indicate = 1 / (self.job.parallel[t_id] + 1)
                            assert self.isc == 'METF'
                            indicate = time_current
                        if max_indicate < indicate and delta_cost < CostToSpend:
                            max_indicate, max_t_id, max_ftype = indicate, t_id, ftype
                if max_indicate == 0:
                    break
                task = self.job.task_list[max_t_id]
                time_current = task.execution_time
                cost_current_inner = time_current / Faas.unit * Faas.prices[selected_type_list[max_t_id]]
                time_new = task.instructions / Faas.speeds[max_ftype] + task.max_data_time
                cost_new = time_new / Faas.unit * Faas.prices[max_ftype]
                task.execution_time = time_new
                CostToSpend = CostToSpend - (cost_new - cost_current_inner)
                selected_type_list[max_t_id] = max_ftype
                total_delay = 0
                for task_inner in self.job.task_list:
                    task_inner.start_time = self.ready_time(task_inner)
                    task_inner.end_time = task_inner.start_time + task_inner.execution_time
                    if total_delay < task_inner.end_time:
                        total_delay = task_inner.end_time
            return CostToSpend, total_delay

        def CriticalPath_Hybrid_Schedule(total_delay, AP, AC):  # AP和AC都在函数外进行修改
            EST, LST = np.zeros(self.job.task_number), np.zeros(self.job.task_number)
            for i in range(self.job.task_number):
                task = self.job.task_list[i]
                EST[i] = task.start_time  # 本文中任务的开始时间就是任务的最早开始时间，当然这两个值并不一定相等
            for i in range(self.job.task_number):
                task = self.job.task_list[self.job.queue[self.job.task_number - i - 1]]
                if not task.children + AC[task.task_id]:  # 这里只需要对 AC 进行修改即可，task.start_time早已确定
                    LST[task.task_id] = total_delay - task.execution_time
                else:
                    latest_finish_time = float("inf")
                    for c in task.children + AC[task.task_id]:
                        child_lst = LST[c]
                        if latest_finish_time > child_lst:
                            latest_finish_time = child_lst
                    LST[task.task_id] = latest_finish_time - task.execution_time
            CriticalTasks, Task_CCR = [], []
            for i in range(self.job.task_number):
                if abs(LST[i] - EST[i]) < 1e-6:
                    CriticalTasks.append(i)
                et = self.job.task_list[i].execution_time
                data_time = self.job.task_list[i].max_data_time
                Task_CCR.append(abs(et - data_time) / et)
            return CriticalTasks, Task_CCR

        def Critical_Allocate_Hybrid_Schedule(CostToSpend, total_delay):
            # 构造增广DAG图和增广DAG矩阵进行关键路径识别
            added_parents, added_children = {}, {}
            for i in range(self.job.task_number):
                added_parents[i] = []
                added_children[i] = []
            # 优先使用VM对关键路径进行优化
            found = True
            self.job.queue = [i for i in range(self.job.task_number)]
            # print("{}:".format(self.rr) + "程序执行前剩余预算为：{}, ".format(CostToSpend) + "总延迟为：{}。".format(total_delay))
            while found:
                VMTasks = []
                for m in self.cluster.machine_running:
                    for task in m.queue:
                        VMTasks.append(task.task_id)
                # selected_type_list 将继续发挥作用，判断所选择的函数类型
                SelectedTasks, SelectedTypes = [], []
                for i in range(self.job.task_number):
                    if (i not in VMTasks) and (Faas.speeds[selected_type_list[i]] <= Machine.speed):
                        SelectedTasks.append(i)
                        SelectedTypes.append(selected_type_list[i])
                if len(SelectedTasks) == 0:  # 如果结果为空，则结束搜索过程
                    break
                # 对所选择的任务按照开始时间排序。然后进行调整，使时间参数与已排序的任务相对应。
                temp_ST, temp_ET, temp_FT = np.zeros(len(SelectedTasks)), np.zeros(len(SelectedTasks)), \
                                            np.zeros(len(SelectedTasks))
                RBIncrease = np.zeros(len(SelectedTasks))
                for i in range(len(SelectedTasks)):
                    index = SelectedTasks[i]
                    task = self.job.task_list[index]
                    temp_ST[i] = task.start_time
                    temp_ET[i] = task.instructions / Machine.speed + task.max_data_time
                    temp_FT[i] = task.end_time
                    RBIncrease[i] = Faas.prices[SelectedTypes[i]] * task.execution_time / Faas.unit - Machine.price * \
                                    temp_ET[i] / Machine.unit
                Temp = np.array([SelectedTasks, list(temp_ST), list(temp_ET), list(temp_FT), RBIncrease,
                                 SelectedTypes]).T.tolist()
                if self.q == 'ESTF':
                    Temp.sort(key=lambda x: x[1])
                elif self.q == 'MRBIF':
                    Temp.sort(key=lambda x: (x[1], x[-4]))  # 优先按照开始时间排序，其次按照函数的类型排序，最后按照执行时间顺序。
                else:
                    assert self.q == 'EFTF'
                    Temp.sort(key=lambda x: x[3])  # 优先按照结束时间排序，其次按照函数的类型排序，最后按照执行时间顺序。
                [SortedSelectedTasks, ST, ET, FT, SortedRBIncrease, SortedSelectedTypes] = np.array(Temp).T.tolist()
                ST, ET, FT = np.array(ST), np.array(ET), np.array(FT)
                SortedSelectedTasks = list(map(int, SortedSelectedTasks))
                found = False
                for i in range(len(SortedSelectedTasks)):
                    # task = self.job.task_list[t_id]
                    t_id = SortedSelectedTasks[i]
                    temp_list = [t_id]  # 首先将任务加入到临时列表中
                    r = Machine()  # 尝试将任务放入新的实例
                    start_time = ST[i]  # 计算r的租赁开始时间，与任务的开始时间相同
                    current_time = FT[i]  # 计算当前时刻，按照结束时间计算
                    Rented_BTUs = np.ceil(ET[i] / r.unit)
                    if self.b == 'MRN':
                        Rented_BTUs = max(Rented_BTUs, 1)
                    elif self.b == 'FRN2':
                        Rented_BTUs = max(Rented_BTUs, 2)
                    elif self.b == 'FRN3':
                        Rented_BTUs = max(Rented_BTUs, 3)
                    elif self.b == 'FRN4':
                        Rented_BTUs = max(Rented_BTUs, 4)
                    else:
                        assert self.b == 'URN'
                        Rented_BTUs = float('inf')
                    rel_time = start_time + Rented_BTUs * r.unit
                    for j in range(i + 1, len(SortedSelectedTasks)):
                        # 如果修改后效果不好，这部分代码可以拿掉。
                        exec_time = ET[j]
                        task = self.job.task_list[SortedSelectedTasks[j]]
                        for k in range(len(temp_list)):
                            if task.transmission.get(temp_list[k]):
                                exec_time = exec_time - task.transmission.get(temp_list[k]) / Task.NetworkBandwidth
                        # 这里，其实FT[j]没有用到。当然，这可以在后续使用。
                        if ST[j] >= current_time and ST[j] + exec_time <= rel_time:  # rel_time 仅在此处有作用。
                            temp_list.append(SortedSelectedTasks[j])  # 注意temp_list中存放的是任务的实际id而非下标。
                            current_time = ST[j] + exec_time  # 没必要对rel_time进行更新，因为代价总会重新计算的。
                        elif (self.q == 'ESTF' or self.q == 'MRBTF') and ST[j] >= rel_time:  # 使用早停判断条件以加速程序
                            break
                    # exitfunc = False
                    # if self.b == "URN":
                    # exitfunc = False
                    added_parents_AF, added_children_AF = {}, {}
                    for k in range(self.job.task_number):
                        added_parents_AF[k] = []
                        added_children_AF[k] = []
                    ET_AF = np.zeros(self.job.task_number)
                    for k in range(self.job.task_number):
                        ET_AF[k] = self.job.task_list[k].execution_time
                    for k in range(len(temp_list)):  # 首先更新task.execution_time
                        task = self.job.task_list[temp_list[k]]
                        data_time = task.max_data_time
                        for s in range(k):
                            if task.transmission.get(temp_list[s]):
                                data_time = data_time - task.transmission.get(temp_list[s]) / Task.NetworkBandwidth
                            else:  # 修改 added_parents 和 added_children
                                parent = temp_list[s]
                                child = temp_list[k]
                                added_parents_AF[child].append(parent)
                                added_children_AF[parent].append(child)
                        ET_AF[temp_list[k]] = task.instructions / Machine.speed + data_time  # 首先更新task.execution_time
                    # 拓扑排序，Kahn算法，保证任务在更新其他时间参数之前，所有前驱的时间参数都被更新
                    current_parents = {}
                    current_children = {}
                    for k in range(self.job.task_number):
                        current_parents[k] = self.job.task_list[k].parents + added_parents_AF[k] + added_parents[k]
                        current_children[k] = self.job.task_list[k].children + added_children_AF[k] + added_children[k]
                    queue, count = [], 0
                    while count < self.job.task_number:
                        for k in range(self.job.task_number):
                            if (current_parents.get(k) is not None) and len(current_parents.get(k)) == 0:
                                queue.append(k)
                                for kc in current_children[k]:
                                    current_parents[kc].remove(k)
                                current_parents.pop(k)
                                current_children.pop(k)
                                count = count + 1
                    # 更新任务的时间参数
                    ST_AF = np.zeros(self.job.task_number)
                    FT_AF = np.zeros(self.job.task_number)
                    for k in range(self.job.task_number):
                        task_inner = self.job.task_list[queue[k]]
                        st = 0
                        for p in task_inner.parents + added_parents_AF[task_inner.task_id] + added_parents[task_inner.task_id]:
                            parent = self.job.task_list[p]
                            if st < FT_AF[parent.task_id]:
                                st = FT_AF[parent.task_id]
                        ST_AF[queue[k]] = st
                        FT_AF[queue[k]] = ST_AF[queue[k]] + ET_AF[queue[k]]
                    cost_old = self.job.Budget - CostToSpend
                    cost_new = 0
                    for k in range(self.job.task_number):
                        if k not in VMTasks and k not in temp_list:
                            cost_new = cost_new + ET_AF[k] * Faas.prices[selected_type_list[k]] / Faas.unit
                    for m in self.cluster.machine_running:
                        cost_new = cost_new + np.ceil((FT_AF[m.queue[-1].task_id] - ST_AF[m.queue[0].task_id])
                                                      / Machine.unit) * Machine.price
                    cost_new = cost_new + np.ceil((FT_AF[temp_list[-1]] - ST_AF[temp_list[0]])
                                                  / Machine.unit) * Machine.price
                    if cost_old >= cost_new:  # 保证成本是下降的
                        # print("{}:".format(self.rr) + "找到成本下降的放置方案！,转移的任务为:{}".format(temp_list))
                        for k in range(len(temp_list)):  # 首先更新task.execution_time
                            task = self.job.task_list[temp_list[k]]
                            exec_time = task.max_data_time
                            for s in range(k):
                                if task.transmission.get(temp_list[s]):
                                    exec_time = exec_time - task.transmission.get(temp_list[s]) / Task.NetworkBandwidth
                                else:  # 修改 added_parents 和 added_children
                                    parent = temp_list[s]
                                    child = temp_list[k]
                                    added_parents[child].append(parent)
                                    added_children[parent].append(child)
                            task.execution_time = task.instructions / Machine.speed + exec_time  # 首先更新task.execution_time
                        self.job.queue = queue
                        # 更新任务的时间参数
                        for k in range(self.job.task_number):
                            task_inner = self.job.task_list[k]
                            task_inner.start_time = ST_AF[k]
                            task_inner.end_time = FT_AF[k]
                        # 将任务加入虚拟机，虚拟机加入集群
                        for k in range(len(temp_list)):
                            task = self.job.task_list[temp_list[k]]
                            r.add_task(task)
                        self.cluster.machine_running.append(r)
                        total_delay = 0
                        # 重新计算剩余预算和总延迟
                        for task_inner in self.job.task_list:
                            if total_delay < task_inner.end_time:
                                total_delay = task_inner.end_time
                        CostToSpend = CostToSpend + cost_old - cost_new
                        found = True
                        # print("调整后总延迟为：" + str(total_delay))
                        # print([CostToSpend, self.job.Budget, total_delay, temp_list])  # 对结果进行校验,保证结果有增长
                    if found:  # or exitfunc:
                        break
            # print("{}:".format(self.rr) + "程序执行后剩余预算为：{}, ".format(CostToSpend) + "总延迟为：{}。".format(total_delay) + "\n")
            return CostToSpend, total_delay, added_parents, added_children
            # 当关键路径无法进一步优化且时，迁移非关键路径上的任务以提升预算余量

        def Critical_ReAllocate(CostToSpend, total_delay, added_parents, added_children):
            if not self.job.queue:
                self.job.queue = [i for i in range(self.job.task_number)]
            while CostToSpend > 0:
                CriticalTasks, Task_CCR = CriticalPath_Hybrid_Schedule(total_delay, added_parents, added_children)
                max_indicate, max_t_id, max_ftype = 0, 0, 0
                VMTasks = []
                for m in self.cluster.machine_running:
                    for task in m.queue:
                        VMTasks.append(task.task_id)

                for t_id in CriticalTasks:
                    if ((self.br == 'PTA') and (t_id not in VMTasks)) or (self.br == 'GTA'):
                        task_inner = self.job.task_list[t_id]
                        time_current = task_inner.execution_time
                        if t_id not in VMTasks:
                            cost_current_inner = time_current / Faas.unit * Faas.prices[
                                selected_type_list[task_inner.task_id]]
                        else:
                            cost_current_inner = time_current / Machine.unit * Machine.price
                            for k in range(len(Faas.speeds)):
                                if Faas.speeds[k] <= Machine.speed:
                                    selected_type_list[t_id] = k
                        # 当CPU资源与价格呈现线性关系时，不需要对类型进行搜索，选择最近的类型即可。
                        if selected_type_list[t_id] + 1 < len(Faas.speeds):
                            ftype = selected_type_list[t_id] + 1
                            time_new = task_inner.instructions / Faas.speeds[ftype] + task_inner.max_data_time
                            cost_new = time_new / Faas.unit * Faas.prices[ftype]

                            delta_time = time_current - time_new
                            delta_cost = cost_new - cost_current_inner
                            if self.isc == 'MTDCIRF':
                                indicate = delta_time / delta_cost
                            elif self.isc == 'MCETRF':
                                indicate = Task_CCR[t_id]
                            else:
                                assert self.isc == 'METF'
                                indicate = time_current
                                # indicate = 1 / (self.job.parallel[t_id] + 1)
                            if max_indicate < indicate and delta_cost < CostToSpend:
                                max_indicate, max_t_id, max_ftype = indicate, t_id, ftype
                if max_indicate == 0:
                    break
                # Map max_t_id to max_ftype
                task = self.job.task_list[max_t_id]
                time_current = task.execution_time
                cost_current_inner = time_current / Faas.unit * Faas.prices[selected_type_list[max_t_id]]
                time_new = task.instructions / Faas.speeds[max_ftype] + task.max_data_time
                cost_new = time_new / Faas.unit * Faas.prices[max_ftype]
                task.execution_time = time_new
                selected_type_list[max_t_id] = max_ftype
                if self.br == 'GTA':  # 完成三个动作： 1、同一机器上后继任务的执行时间更新， 2、修改约束关系， 3、将任务从虚拟机中剔除。
                    for m in self.cluster.machine_running:
                        if task in m.queue:
                            m.queue.remove(task)
                            VMTasks.remove(task.task_id)
                            for child in task.children:
                                c_task = self.job.task_list[child]
                                if c_task in m.queue:
                                    c_task.execution_time += c_task.transmission[task.task_id] / Task.NetworkBandwidth
                            added_parents[task.task_id] = []
                            added_children[task.task_id] = []
                            break
                delay = 0
                for t_id in self.job.queue:  # queue在任何情况下都不需要重新计算。
                    task_inner = self.job.task_list[t_id]
                    for p in task_inner.parents + added_parents[task_inner.task_id]:
                        parent = self.job.task_list[p]
                        if task_inner.start_time < parent.end_time:
                            task_inner.start_time = parent.end_time
                    task_inner.end_time = task_inner.start_time + task_inner.execution_time  # 这里有错误？
                    if delay < task_inner.end_time:
                        delay = task_inner.end_time
                total_cost = 0
                for m in self.cluster.machine_running:
                    if len(m.queue) > 0:
                        m.cost = np.ceil((m.queue[-1].end_time - m.queue[0].start_time) / Machine.unit) * Machine.price
                    else:
                        m.cost = 0
                    total_cost = total_cost + m.cost
                for i in range(self.job.task_number):
                    if i not in VMTasks:
                        task_inner = self.job.task_list[i]
                        cost_incre = task_inner.execution_time / Faas.unit * Faas.prices[selected_type_list[i]]
                        total_cost = total_cost + cost_incre
                if total_cost > self.job.Budget:  # 这里必须设置，因为任务大小调整之后可能会影响到虚拟机上的任务跨度
                    break
                total_delay = delay
                CostToSpend = self.job.Budget - total_cost
                # CostToSpend = CostToSpend - (cost_new - cost_current_inner)
            return CostToSpend, total_delay

        # 首先将所有任务分配到最便宜的函数上面去。
        selected_type_list = [0 for i in range(self.job.task_number)]
        for t in self.job.task_list:
            selected_type = selected_type_list[t.task_id]
            t.execution_time = t.instructions / Faas.speeds[selected_type] + t.max_data_time
            t.start_time = self.ready_time(t)
            t.end_time = t.start_time + t.execution_time
        # 计算首次分配后的end-to-end delay 和 cost_current
        end2end_delay, cost_current = 0, 0
        for t in self.job.task_list:
            if end2end_delay < t.end_time:
                end2end_delay = t.end_time
            cost_current = cost_current + t.execution_time / Faas.unit * Faas.prices[selected_type_list[t.task_id]]
        # 对首次分配后的结果进行调整，优先使用高性能函数进行调整。
        cost_checkpoint, makespan_checkpoint = self.job.Budget - cost_current, end2end_delay
        if self.G1:
            cost_checkpoint, makespan_checkpoint = Critical_Allocate(cost_checkpoint, makespan_checkpoint)
        # return cost_checkpoint, makespan_checkpoint
        # 对首次分配后的结果进行调整，其次使用VM实例进行调整。
        ap, ac = {}, {}
        for i in range(self.job.task_number):
            ap[i] = []
            ac[i] = []
        if self.G2:
            cost_checkpoint, makespan_checkpoint, ap, ac = Critical_Allocate_Hybrid_Schedule(cost_checkpoint,
                                                                                             makespan_checkpoint)
        # 将节省下来的预算重新用于加速关键路径上的任务
        if self.G3:
            cost_checkpoint, makespan_checkpoint = Critical_ReAllocate(cost_checkpoint, makespan_checkpoint, ap, ac)
        return cost_checkpoint, makespan_checkpoint
