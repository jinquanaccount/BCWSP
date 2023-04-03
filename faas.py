

class Faas:
    prices = [0.01234, 0.02469, 0.04937, 0.09874, 0.19748, 0.39496, 0.78992]
    speeds = [0.25, 0.5, 1, 2, 4, 8, 16]  # 平均值为4.5
    unit = 3600

    def __init__(self, Type):
        self.start_time = 0
        self.end_time = 0
        self.task = None               # 分配给该函数的任务是哪一个
        self.cost = 0                  # 函数的总费用
        self.type = Type
        self.price = Faas.prices[self.type]
        self.speed = Faas.speeds[self.type]

    def add_task(self, task):          # 要求任务中已经填入信息
        self.start_time = task.start_time
        self.end_time = task.end_time
        self.task = task
        self.cost = (self.end_time - self.start_time) * self.price / Faas.unit


