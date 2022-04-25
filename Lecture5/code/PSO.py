# %% [markdown]
# # 智能算法求解优化问题
# ## 课堂练习
#
# <div align=center><img src='./pics/pic14.png' /></div>
#
# $$
# S=\left[
# \begin{matrix}
# 0 & -7/12 & -1/4 \\
# 0 & -5/12 & -3/4 \\
# 0 & 5/12 & -1/4
# \end{matrix}
# \right]
# $$
#
# 目标函数
# $$
# min:
# \begin{cases}
# F &= 0.0007P_{G1}^2+0.30P_{G1}+4 \\
#   &+ 0.0004P_{G2}^2+0.32P_{G2}+3
# \end{cases}
# $$
# 运行约束：
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
# %% [markdown]
# # 考虑发电机阀点效应的最优潮流计算
# <div align=center><img src='./pics/pic17.png' /></div>
#
# $$
# min:
# \begin{cases}
# F&=0.0007P_{G1}^2+0.30P_{G1}+4+\left|10*sin(0.126 * (30-P_{G1}))\right| \\
# &+0.0004P_{G2}^2+0.32P_{G2}+3+\left|3*sin(0.378 * (0-P_{G2}))\right|
# \end{cases}
# $$
# %% [markdown]
# # 多峰优化问题
# <div align=center><img src='./pics/pic18.png' /></div>
#
# ## 目标函数
# $$
# min:
# \begin{cases}
# F&=0.0007P_{G1}^2+0.30P_{G1}+4+\left|10*sin(0.126 * (30-P_{G1}))\right| \\
# &+0.0004P_{G2}^2+0.32P_{G2}+3+\left|3*sin(0.378 * (0-P_{G2}))\right|
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
# 传统数学方法难以求解的生产调度问题，可以借助人工智能的方法——智能计算
# %% [markdown]
# ## 基本概念
# 计算智能主要包括：
#
# 1. **进化计算**、**群智能计算**
# <div align=center><img src='./pics/pic19.png' /></div>
# <p align=center>https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=4235</p>
#
# 2. **神经计算**
# <div align=center><img src='./pics/pic20.png' /></div>
# <p align=center>https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385</p>
#
# 3. **模糊计算**
# <div align=center><img src='./pics/pic21.png' /></div>
# <p align=center>https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=91</p>
# %% [markdown]
# # 测试函数库
#
# **高维单模函数**：只有一个极值点，决策变量多
# <div align=center><img src='./pics/pic22.png' /></div>
#
# **高维多模函数**：多极值点，决策变量多
# <div align=center><img src='/pics/pic23.png' /></div>
#
# **低维多模函数**：多极值点，决策变量少
# <div align=center><img src='./pics/pic24.png' /></div>
#
# %% [markdown]
# # 测试函数库——举例
# <div align=center><img src='./pics/pic25.png' /></div>
#
# <div align=center><img src='./pics/pic26.png' /></div>
#
# %% [markdown]
# # 群体智能
# + 群体智能源于对以*蚂蚁*、*蜜蜂*等为代表的*社会性昆虫*的*群体行为*的研究。最早被用在细胞机器人系统的描述中。
# + 1991年意大利学者Dorigo提出蚁群优化（Ant Colony Optimization，ACO）理论，群体智能作为一个理论被正式提出，并逐渐吸引了大批学者的关注，从而掀起了研究高潮
# + 1995年，Kennedy 等学者提出粒子群优化算法（Particle Swarm Optimization，PSO)，此后群体智能研究迅速展开，但大部分工作都是围绕ACO和PSO 进行的。
# %% [markdown]
# # 粒子群优化算法(Particle Swarm Optimization, PSO)
# + 由**James Kenney（社会心理学博士）**肯尼迪和**Russ Eberhart（电子工程学博士）**艾伯哈特，1995年提出；
# + 模拟鸟群或蜂群的觅食行为；
# + 基本思想：通过群体中个体之间的协作和信息共享来寻找最优解。
# %% [markdown]
# # 鸟类的觅食
# + 假设一群鸟在随机的搜索食物，在一块区域里只有一块食物，所有的鸟都不知道食物在哪。但是它们知道自己的当前位置距离食物有多远。
# + 那么这群鸟找到食物的最优策略是什么？
#
# **最简单有效的就是搜寻离食物最近的鸟的周围区域**
# %% [markdown]
# # 粒子群优化算法
# ## PSO算法基本思路
# + 初始化为一群随机粒子（位置和速度），通过迭代找到最优。
# + 每次迭代中，粒子通过跟踪“个体极值(pbest)”和“全局极值(gbest) ”来更新自己的位置。
# <div align=center><img src='./pics/pic27.png' /></div>
#
# %% [markdown]
# # 粒子群优化算法——关键公式
# ## 粒子速度和位置的更新
# $$
# v_{id}^{k+1} = wv_{id}^k + c_1rand()(p_{id}-x_{id}^k)+c_2rand()(p_{gbest}-x_{id}^k)
# $$
# $$
# x_{id}^{k+1}=x_{id}^k+v_{id}^{k+1} \quad i=1,2,\cdots ,m;\quad d=1,2,\cdots ,D
# $$
#
# + 其中，w称为惯性权重，
# + c1和c2为两个大于0的常系数，称为加速因子。
# + rand()为\[0,1\]上的随机均值函数。
# + xd为粒子当前位置，pd为粒子历史最好位置，pgbest为全体粒子所经过的最好位置，vd为粒子速度
#
# <div align=center><img src='./pics/pic28.png' /></div>
#
# %% [markdown]
# # 粒子群优化算法——关键点
# ## 惯性权重w
# + 表示微粒对当前自身运动状态的信任，依据自身的速度进行惯性运动，使其有扩展搜索空间的趋势，有能力探索新的区域。
# + 较大的w有利于跳出局部极值，而较小的w有利于算法收敛。
# $$
# v_{id}^{k+1} = wv_{id}^k + c_1rand()(p_{id}-x_{id}^k)+c_2rand()(p_{gbest}-x_{id}^k)
# $$
# $$
# x_{id}^{k+1}=x_{id}^k+v_{id}^{k+1} \quad i=1,2,\cdots ,m;\quad d=1,2,\cdots ,D
# $$
# %% [markdown]
# # 粒子群优化算法——关键点
# ## 加速常数$c_1$和$c_2$
# + 代表将粒子推向pbest和gbest位置的统计加速项的权重。
# + 较小的值允许粒子在被拉回之前可以在目标区域外徘徊，而较大的值则导致粒子突然冲向或越过目标区域。
# + 将$c_1$和$c_2$统一为一个控制参数，$\phi= c_1+c_2$
# + 如果$\phi$很小，粒子群运动轨迹将非常缓慢；
# + 如果$\phi$很大，则粒子位置变化非常快；
# + 实验表明，当$\phi=4$（通常$c_1=2.0$，$c_2=2.0$）时，具有很好的收敛效果。
#
# $$
# v_{id}^{k+1} = wv_{id}^k + c_1rand()(p_{id}-x_{id}^k)+c_2rand()(p_{gbest}-x_{id}^k)
# $$
# $$
# x_{id}^{k+1}=x_{id}^k+v_{id}^{k+1} \quad i=1,2,\cdots ,m;\quad d=1,2,\cdots ,D
# $$

# %% [markdown]
# # 粒子群优化算法——高维空间
# ## 粒子速度和位置-符号表达：
# + 假设在D维搜索空间中，有m个粒子；
# + 其中第i个粒子的位置为矢量$\vec{x_i}=(x_{i1},x_{i2},\cdots ,x_{iD})$
# + 其飞翔速度也是一个矢量，记为$\vec{v_i}=(v_{i1},v_{i2},\cdots ,v_{iD})$
# + 第i个粒子搜索到的最优位置为$\vec{p_i}=(p_{i1},p_{i2},\cdots ,p_{iD})$
# + 整个粒子群搜索到的最优位置为$\vec{p_{gbest}}=(p_{gbest1},p_{gbest2},\cdots ,p_{gbestD})$
# %% [markdown]
# # 粒子群优化算法流程
# 1. 初始化一群粒子（群体规模），包括随机的位置和速度；
# 2. 评价每个粒子的适应度；
# 3. 对每个粒子更新个体最优位置；
# 4. 更新全局最优位置；
# 5. 根据速度和位置方程更新每个粒子的速度和位置；
# 6. 如未满足结束条件（通常为满足足够好的适应值或达到设定的最大迭代次数），返回②。
# %% [markdown]
# # 粒子群优化算法——其他注意事项
# ## 粒子数
# 一般取20～40，对较难或特定类别的问题可以取100～200。
# ## 最大速度$v_{max}$
# 决定粒子在一个循环中最大的移动距离，通常设定为粒子的范围宽度。
# ## 终止条件
# 最大循环数以及最小错误要求。
# %% [markdown]
# # 课堂练习
#
# **高维单模函数**：只有一个极值点，决策变量多
# <div align=center><img src='./pics/pic22.png' /></div>
#
# **高维多模函数**：多极值点，决策变量多
# <div align=center><img src='./pics/pic23.png' /></div>
#
# **低维多模函数**：多极值点，决策变量少
# <div align=center><img src='./pics/pic24.png' /></div>
#
# %% [markdown]
# # 课堂练习
# <div align=center><img src='./pics/pic18.png' /></div>
#
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
from sko.PSO import PSO
import numpy as np
import matplotlib.pyplot as plt

# 加速常数
c1 = 2
c2 = 2

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

max_iter = 500  # 迭代次数
num_particle = 40  # 粒子数目

# 惯性权重随迭代次数增加而逐渐减小，有利于算法收敛
start_weight = 0.9
end_weight = 0.4
weight_step = (start_weight - end_weight) / max_iter

# 目标函数，仅包含pg1一个自变量


def objective(x): return 0.0007 * x ** 2 + 0.3 * x + 4 + \
    0.0004 * (100 - x) ** 2 + 0.32 * (100 - x) + 3


# 初始化粒子群，每个粒子在定义域内随机选择一个初始位置
particle = xmin + (xmax - xmin) * np.random.rand(num_particle, NDim)

# 初始速度
V = 0.5 * (xmax - xmin) * (np.random.rand(num_particle, NDim)-0.5)

# 粒子适应度
fitness = np.zeros((num_particle, NDim))
# 粒子最佳位置
pbest = np.zeros((num_particle, NDim))

# 计算初始适应度
for i in range(num_particle):
    pg1 = particle[i, :]
    pg2 = 100 - pg1
    if pg2Min <= pg2 <= pg2Max:
        tmp_array = np.vstack((pg1, pg2, -100))
        if (np.dot(S, tmp_array) >= -flim).all() and (np.dot(S, tmp_array) <= flim).all():
            fitness[i] = objective(pg1)
        else:
            # 若pg1和pg2不满足潮流上下限约束则适应度为无穷
            fitness[i] = np.inf
    else:
        # 若pg2不满足给定出力上下限约束则适应度为无穷
        fitness[i] = np.inf

for i in range(num_particle):
    pbest[i, :] = particle[i, :]

# 每个粒子的当前最佳适应度
pbest_value = fitness
# 当前的最佳适应度
gbest_value = np.min(pbest_value)
# 当前最佳适应度粒子的位置
gbest = particle[np.argmin(pbest_value)]

# 记录每次迭代的最佳目标函数值
obj_fun_val = np.zeros((max_iter, 1))

for iter in range(max_iter):
    # 惯性权重
    w = start_weight - (iter + 1) * weight_step
    # 速度
    V = w * V + c1 * np.random.rand() * (pbest - particle) + c2 * \
        np.random.rand() * (gbest - particle)  # 更新速度
    # 更新粒子位置
    particle += V
    # 将粒子位置约束在xmin和xmax之间
    particle = np.clip(particle, xmin, xmax)

    # 计算适应度
    for i in range(num_particle):
        pg1 = particle[i]
        pg2 = 100 - pg1
        if 0 <= pg2 <= 50:
            tmp_array = np.vstack((pg1, pg2, -100))
            if (np.dot(S, tmp_array) >= -flim).all() and (np.dot(S, tmp_array) <= flim).all():
                # 满足约束
                fitness[i] = objective(pg1)
            else:
                # 若pg1和pg2不满足潮流上下限约束则适应度为无穷
                fitness[i] = np.inf
        else:
            # 若pg2不满足给定出力上下限约束则适应度为无穷
            fitness[i] = np.inf

    # 更新粒子位置和最佳适应度
    pbest_value = np.minimum(pbest_value, fitness)

    # 更新每个粒子取得最佳适应度的位置
    for i in range(num_particle):
        if pbest_value[i] == fitness[i]:
            pbest[i] = particle[i]

    # 本次迭代的最佳适应度
    current_min_value = np.min(pbest_value)
    # 总体最佳适应度
    gbest_value = np.minimum(current_min_value, gbest_value)
    if gbest_value == current_min_value:
        # 总体粒子最佳位置
        gbest = pbest[np.argmin(pbest_value)]

    obj_fun_val[iter] = gbest_value

print('PG1 =', gbest, 'PG2 =', 100 - gbest, 'F =', obj_fun_val[-1])

plt.plot(np.arange(0, max_iter), obj_fun_val)
plt.xlabel('Iteration count')
plt.ylabel('Objective function value')
plt.show()
# %%
# scikit-opt文档比较全，作者是蚂蚁算法工程师，允许自定义算子，支持GPU加速等功能
# 但是用户反映优化有点问题
# PSO暂不支持等式约束

S = np.array([
    [0, -7 / 12, -1 / 4],
    [0, -5 / 12, -3 / 4],
    [0, 5 / 12, -1 / 4]
])
flim = [30, 80, 50]


def objective(x): return 0.0007 * x ** 2 + 0.3 * x + 4 + \
    0.0004 * (100 - x) ** 2 + 0.32 * (100 - x) + 3


# -flim <= S * [[pg1], [pg2], [-100]] <= flim, 0 <= pg2 <= 50
constraint_ueq = [lambda x: -S[i, 0] * x - S[i, 1] * (100 - x) - S[i, 2] * (-100) - flim[i] for i in range(3)] + \
                 [lambda x: S[i, 0] * x + S[i, 1] * (100 - x) + S[i, 2] * (-100) - flim[i] for i in range(3)] + \
                 [lambda x: 100 - x - 50, lambda x: x - 100]

pso = PSO(func=objective, n_dim=1, pop=40, max_iter=500, lb=[30], ub=[
          150], w=0.9, c1=2, c2=2, constraint_ueq=constraint_ueq)
pso.run()
print('PG1 =', pso.gbest_x, 'PG2 =', 100 - pso.gbest_x, 'F =', pso.gbest_y)
plt.plot(pso.gbest_y_hist)
plt.show()
# %% [markdown]
# # 课堂练习
# <div align=center><img src='../../pics/pic18.png' /></div>
#
# ## 目标函数
# $$
# min:
# \begin{cases}
# F&=0.0007P_{G1}^2+0.30P_{G1}+4+\left|10*sin(0.126 * (30-P_{G1}))\right| \\
# &+0.0004P_{G2}^2+0.32P_{G2}+3+\left|3*sin(0.378 * (0-P_{G2}))\right|
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
# %% [markdown]
# # 小结
# ## 优点
#
# + 可以适应较为复杂的优化问题求解，弱化了对于优化问题数学模型性质的要求。
#
# + 编程相对容易实现。
#
# + 隐含并行性。
#
#
# ## 缺点
#
# + 相比于传统优化算法或经典求解算法，智能优化计算的数学理论基础相对薄弱，涉及的各种参数设置没有确切的理论依据。
#
# + 带有随机性，每次的求解不一定一样，当处理突发事件时，系统的反映可能是不可预测的，这在一定程度上增加了其应用风险。
#
# + 高维大规模优化问题求解速度较慢，甚至不收敛，难以适应一些对计算速度要求较高的应用场景。
#
# + 对于某些等式约束的处理可能会比较麻烦。
# %%
