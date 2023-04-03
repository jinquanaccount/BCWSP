import math
import numpy as np


class Machine(object):
    price = None
    speed = None
    unit = 3600

    prices = [0.0255, 0.051, 0.102, 0.204]
    speeds = [1, 2, 4, 8]

    def __init__(self):
        self.queue = []  # 任务队列
        self.start_time = 0
        self.end_time = 0
        self.cost = 0

    def add_task(self, task):
        if len(self.queue) == 0:  # 这里要求task已经填充了信息
            self.start_time = task.start_time
        self.queue.append(task)
        self.end_time = self.start_time + math.ceil((self.avail_time()-self.start_time) / Machine.unit) * Machine.unit
        self.cost = (self.end_time - self.start_time) * self.price / Machine.unit

    def avail_time(self):
        if len(self.queue) == 0:
            return 0
        else:
            return self.queue[-1].end_time

    def insert_task(self, task):  # 在已经分配了一部分任务时使用
        # self.queue.append(task)
        self.queue = self.queue + task  # 这里的任务是一个列表
        self.queue.sort(key=lambda t: t.start_time)  # 重新排序
        self.start_time = self.queue[0].start_time
        self.end_time = self.start_time + math.ceil((self.avail_time() -
                                                     self.start_time) / Machine.unit) * Machine.unit
        self.cost = (self.end_time - self.start_time) * self.price / Machine.unit


if __name__ == '__main__':
    m = Machine()
    print(m.price)
