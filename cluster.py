

class Cluster:
    def __init__(self):
        self.faas_pool = []
        self.machine_running = []
        self.machine_unused = []

    def add_VM(self, VM):
        self.machine_running.append(VM)

    def add_Function(self, F):
        self.faas_pool.append(F)
