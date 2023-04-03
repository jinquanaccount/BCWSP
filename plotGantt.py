
import matplotlib.pyplot as plt


def gantt(best_solution):
    functions = best_solution[0]
    machines = best_solution[1]
    for i in range(len(machines)):
        machine = machines[i]
        for task in machine.queue:
            plt.barh("Vm" + str(i), task.end_time - task.start_time - 1, left=task.start_time)
            plt.text(task.start_time, "Vm" + str(i), 'T%s' % task.task_id, color="black")
    for i in range(len(machines), len(machines)+len(functions)):
        function = functions[i-len(machines)]
        task = function.task
        plt.barh("F" + str(i), task.end_time - task.start_time - 1, left=task.start_time)
        plt.text(task.start_time, "F" + str(i), 'T%s' % task.task_id, color="black")
    plt.title('Resources & tasks mapping Gantt-chart with Cost %s' % round(best_solution[2], 3), fontsize=16, color='r')
    plt.xlabel('Time')
    plt.ylabel('Resource')
    plt.show()
