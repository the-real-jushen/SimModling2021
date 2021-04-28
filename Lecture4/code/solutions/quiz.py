# %% [markdown]
'''
# quiz 1, 应用2
# ![avatar](../pics/pic3.png)
'''

# %% [markdown]
# 目标函数 $max z = 5.24x_1 + 7.30x_2 + 8.34x_3 + 4.18x_4$
#
# 约束条件
# $$
# \begin{cases}
# 1.5x_1 + 1.0x_2 + 2.4x_3 + 1.0x_4 \le 2000 \\
# 1.0x_1 + 5.0x_2 + 1.0x_3 + 3.5x_4 \le 8000 \\
# 1.5x_1 + 3.0x_2 + 3.5x_3 + 1.0x_4 \le 5000 \\
# x_1, x_2, x_3, x_4 \ge 0 \\
# \end{cases}
# $$
# 这里我们忽略整数限制条件
# %%

from scipy.optimize import linprog
import numpy as np
c = [-5.24, -7.3, -8.34, -4.18]
A_ub = [[1.5, 1, 2.4, 1],
        [1, 5, 1, 3.5],
        [1.5, 3, 3.5, 1]]
b_ub = [2000, 8000, 5000]

res = linprog(c, A_ub, b_ub)
print(res)
# %%

# %% [markdown]
'''
# quiz 2, 应用3
![avatar](./pics/pic4.png)
'''

# %%

model = pulp.LpProblem('应用3', pulp.LpMaximize)
V_NUM = 2

x = [pulp.LpVariable(
    f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(V_NUM)]
# 这里可以把约束条件和目标系数写成矩阵的形式，然后用求和来写表达式
c = [40, 90]
objective = sum(c[i] * x[i] for i in range(V_NUM))

constraints = []
a_cons = ([9, 7], [7, 20])
b_cons = [56, 70]
# enumerate，除了没每一个元素还给出它的index
for idx, a in enumerate(a_cons):
    constraints.append(sum([a[i] * x[i] for i in range(V_NUM)]) <= b_cons[idx])
# 添加目标个条件
model += objective
for cons in constraints:
    model += cons

# 求解模型
status = model.solve()

# 打印出结果
for xi in model.variables():
    print(xi.name, ' = ', xi.varValue)
print("objective value: ", model.objective.value())

# %% [markdown]
'''
# quiz 3, 应用4
![avatar](./pics/pic5.png)
'''

# %%
model = pulp.LpProblem('应用4', pulp.LpMinimize)
V_NUM = 5
x = [pulp.LpVariable(
    f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(V_NUM)]
# 这里可以把约束条件和目标系数写成矩阵的形式，然后用求和来写表达式
c = [0, 0.1, 0.2, 0.3, 0.8]
objective = sum(c[i] * x[i] for i in range(V_NUM))

constraints = []
a_cons = ([1, 2, 0, 1, 0], [0, 0, 2, 2, 1], [3, 1, 2, 0, 3])
b_cons = 100
# enumerate，除了没每一个元素还给出它的index
for idx, a in enumerate(a_cons):
    constraints.append(sum([a[i] * x[i] for i in range(V_NUM)]) == b_cons)
# 添加目标个条件
model += objective
for cons in constraints:
    model += cons

# 求解模型
status = model.solve()

# 打印出结果
for xi in model.variables():
    print(xi.name, ' = ', xi.varValue)
print("objective value: ", model.objective.value())
