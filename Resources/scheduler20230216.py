import copy
from machine import Machine
from faas import Faas
from task import Task
import numpy as np


class Scheduler:
    def __init__(self, job, cluster, G1, G2, G3, isc, q, b, br, price, speed):
        self.job_bak = job
        self.cluster_bak = cluster
        self.job = None
        self.cluster = None
        self.money = 0
        self.makespan = 0

        self.G1 = G1
        self.G2 = G2
        self.G3 = G3
        self.isc = isc
        self.q = q
        self.b = b
        self.br = br
        Machine.price = price
        Machine.speed = speed

    def run(self):
        self.job = copy.deepcopy(self.job_bak)
        self.cluster = copy.deepcopy(self.cluster_bak)
        RemainingCost, self.makespan = self.BCWS()
        return self.makespan

    def ready_time(self, task):
        ready = 0
        for p in task.parents:
            parent = self.job.task_list[p]
            if ready < parent.end_time:
                ready = parent.end_time
        return ready

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
                temp_ST, temp_ET, temp_FT = np.zeros(len(SelectedTasks)), np.zeros(len(SelectedTasks)),\
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
                    if found: # or exitfunc:
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
                                    c_task.execution_time += c_task.transmission[task.task_id]/Task.NetworkBandwidth
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
