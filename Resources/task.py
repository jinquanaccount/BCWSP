import numpy as np
np.random.seed(2)
from faas import Faas


class Task(object):
    NetworkBandwidth = 10 * pow(2, 20)

    def __init__(self, job, task_id, children, parents, instructions, total_input_size, total_output_size):
        self.job = job
        self.task_id = task_id
        self.instructions = instructions*np.average(Faas.speeds)  # 此处进行了改动，要特别注意
        self.total_input_size = total_input_size
        self.total_output_size = total_output_size
        self.children = children
        self.parents = parents
        self.transmission = None
        self.max_data_time = (total_input_size + total_output_size)/Task.NetworkBandwidth
        self.execution_time = 0

        self.start_time = 0
        self.end_time = 0
        self.scheduled = False
        self.priority = 0
        self.level = 0
