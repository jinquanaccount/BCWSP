import pandas as pd
import numpy as np
data_reader = pd.read_csv("../Parameter_Calibration20230216.csv", usecols=["Makespan"])
Cost = data_reader["Makespan"]
result = np.zeros(len(Cost))
instances = 200  # 注意这里的instance不是文件的数目，而是影响因素的组合数。
comparison = 90
for i in range(instances):
    min_value = float("inf")
    for j in range(comparison):
        if min_value > float(Cost[comparison*i+j]):
            min_value = float(Cost[comparison*i+j])
    for j in range(comparison):
        result[comparison*i+j] = (float(Cost[comparison*i+j]) - min_value)/min_value * 100
with open("RPD.txt", "a+", encoding='utf-8') as f:  # 保存结果
    f.truncate(0)
    for r in result:
        f.write(str(r))
        f.write('\r')
