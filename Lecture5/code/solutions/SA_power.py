# %% [markdown]
# # 课堂练习

# ## 目标函数
# $$
# min:
# \begin{cases}
# F &= 0.0007P_{G1}^2+0.30P_{G1}+4 \\
#   &+ 0.0004P_{G2}^2+0.32P_{G2}+3
# \end{cases}
# $$
#
# ## 运行约束
# 1. 节点功率平衡方程
# $$ P_{G1} + P_{G2} = 100 $$
# 2. 发电机出力上下限
# $$ 30 \le P_{G1} \le 150 $$
# $$ 0 \le P_{G2} \le 50 $$
# 3. 线路潮流上下限
# $$
# -\left[
# \begin{matrix}
# 30 \\
# 80 \\
# 50 \\
# \end{matrix}
# \right]
# \le S
# \left[
# \begin{matrix}
# P_{G1} \\
# P_{G2} \\
# -100 \\
# \end{matrix}
# \right]
# \le
# \left[
# \begin{matrix}
# 30 \\
# 80 \\
# 50 \\
# \end{matrix}
# \right]
# $$
#
# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import savez_compressed

S = np.array([
    [0, -7 / 12, -1 / 4],
    [0, -5 / 12, -3 / 4],
    [0, 5 / 12, -1 / 4]
])
flim = np.array([30, 80, 50]).reshape(3, 1)

# PG1为自变量，通过PG1+PG2=100约束获得PG2
xmin = 30
xmax = 150
pg2Min = 0
pg2Max = 50
NDim = 1

# 目标函数，仅包含pg1一个自变量


def objective(x): return 0.0007 * x ** 2 + 0.3 * x + 4 + \
    0.0004 * (100 - x) ** 2 + 0.32 * (100 - x) + 3


# %%
# simulated annealing
# random solution for Pg1


def initial_solution():
    return np.random.randint(120)+50

# 对一个解进行微调，判断是否满足约束条件


def random_move(solution):
    newSolution = 0
    while(1):
        newSolution = solution-(0.5+np.random.randn())*2
        pg1 = newSolution
        pg2 = 100 - pg1
        if pg2Min <= pg2 <= pg2Max:
            tmp_array = np.vstack((pg1, pg2, -100))
            if (np.dot(S, tmp_array) >= -flim).all() and (np.dot(S, tmp_array) <= flim).all():
                return newSolution


def calc_energy(solution):
    score = objective(solution)
    return score

# 计算接受概率


def probability(delta, T):
    return np.exp(-delta / T)

# 检查是否接收新的解


def deal(x1, x2, delta, T):
    # Delta < 0直接接受，
    if delta < 0:
        return x2, True
    # Delta> 0依概率接受
    p = probability(delta, T)
    if p > np.random.random():
        return x2, True
    return x1, False


def print_status(trial, accept, best):
    print('Trial:', trial, 'Accept:', accept, 'Accept Rate:', '%.2f' %
          (accept / trial), 'Best:', best)


# %%


# 初始温度
Tmax = 1
# 终止温度
Tmin = 0.1
# 温度下降率
rate = 0.8
# 每个温度迭代次数
length = 10000

T = Tmax

# 初始化解
solution = initial_solution()
# 保存当前最优解的分数
best_energy = calc_energy(solution)
# 保存当前最优解
best_solution = solution

loop_count = 0
trial, accept = 0, 0


while T >= Tmin:
    for i in range(length):
        energy = calc_energy(solution)
        # 更新当前最优解
        if best_energy > energy:
            best_energy = energy
            best_solution = solution
            # # 已经得到最优解，提前退出
            # if best_energy == -162:
            #     break
        # 对上一个解做个随机扰动
        random_solution = random_move(solution)
        # 计算这个解的好坏，和上一个解对比
        random_energy = calc_energy(random_solution)
        delta = random_energy - energy
        # 决定时候接受这个新的解或者保持上一个解
        solution, accepted = deal(solution, random_solution, delta, T)

        if accepted:
            accept += 1
        # 记个数，看看试了多少次
        trial += 1

    # 降低温度
    T *= rate
    loop_count += 1
    if loop_count % 1 == 0:
        print_status(trial, accept, best_energy)


print(best_solution)
