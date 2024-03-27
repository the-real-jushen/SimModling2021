# %% [markdown]
# # 遗传算法(Genetic Algorithms, GA)
# ## 基本思想
# 基于自然选择和基因遗传学原理，借鉴了生物进化优胜劣汰的自然选择机理和生物界繁衍进化的基因重组、突变的遗传机制的全局自适应概率搜索算法。
#
# ## 基本思路
# + 初始化一组随机产生的初始解（种群），这个种群由经过基因编码的一定数量的个体组成，每个个体实际上是染色体带有特征的实体。
#
# + 每次迭代根据基因编码计算每个个体的适应度，判断是否达到终止条件。没达到终止条件则淘汰一部分个体，将剩余个体交叉变异形成一组新种群
#
# 重复上述步骤。
#
#![](pics/pic30.png)

# %% [markdown]
# ## 基本套路
# ### 染色体编码
# 利用遗传算法求解问题时必须在目标问题的实际表示与染色体位串结构之间建立一个联系。
#
# 对于给定的优化问题，由种群个体的表现型集合组成的空间称为问题空间，由种群基因型个体组成的空间称为编码空间。由问题空间向编码空间的映射称为编码，由编码空间向问题空间的映射称为解码。
# #### 常用编码方式
# 常用的编码方式有两种：二进制编码和浮点数（实数）编码
# ##### 二进制编码
# 二进制编码是遗传算法最常用的编码方法，它将问题空间的参数用字符集\{1,0\}构成染色体位串，符合最小字符集原则，便于用模式定理分析，但存在映射误差。
#
# 二进制编码方式的编码和解码简单易行，使得遗传算法的交叉和变异操作实现方便。但是当连续函数离散化时存在映射误差。此外，优化问题所需的精度越高对编码串长度的需求也会相应增加，这就导致搜索空间
# 急剧增大，计算量和求解时间也相应增加
#
# ##### 浮点数（实数）编码
# 浮点数编码能够解决二进制编码的这些问题。该方法中个体的每个基因都采用参数给定范围区间的某个浮点数表示，编码长度则取决于决策变量的总数。交叉、变异等操作也要保证新个体的基因值在这个范围内。
# %% [markdown]
#
# ### 适应度函数
# 适应度函数是衡量个体优劣、度量个体适应度的函数。适应度函数值越大则个体越好。遗传算法中根据适应值对个体进行选择，一般适应度函数是由目标函数变换而来。
#
# 由于遗传算法中根据适应度排序的情况计算选择概率，因此适应度函数值不能小于零。此外，将目标函数转换为最大化问题形式且函数值非负的适应度函数是必要的。
# %% [markdown]
# ### 约束条件的处理
# 遗传算法中必须对约束条件进行处理。一般有以下三种方法：
# #### 罚函数法
# 罚函数法的基本思想是，对于解空间中无对应可行解的个体，计算其适应度时除以一个罚函数，从而降低该个体的适应度，降低该个体被遗传到下一代的概率。
# $$
# F'(x)=
# \begin{cases}
# F(x), &x\in U \\
# F(x) - P(x), & x\notin U
# \end{cases}
# $$
# #### 搜索空间限定法
# 对遗传算法的搜索空间大小加以限制，使得搜索空间中表示一个个体的点与解空间中表示一个可行解的点有一一对应的关系。
# #### 可行解变换法
# 在由个体基因型到个体表现型的变换中，增加使其满足的约束条件的处理过程，其寻找个体基因型与个体表现型的多对一变换关系，扩大
# 搜索空间，使进化过程中产生的个体总能通过这个变幻转换成解空间满足约束条件的一个可行解
# %% [markdown]
'''
 ### 遗传算子
 遗传算法包含了三个模拟生物基因遗传操作的遗传算子：选择、交叉和变异
 #### 选择操作
 选择操作用来确定如何从附带群体按某种方法选取个体遗传到下一代群体，适应度较高的个体遗传到下一代的概率较大。

 常用的选择方法有轮盘赌法、排序选择法和两两竞争法。
 ##### 轮盘赌法
 以第i个个体入选种群的概率以及群体规模的上限确定其生存与淘汰。
 1. 计算各染色体$v_k$的适应值$F(v_k)$

 2. 计算种群中所有染色体的适应值之和$Fall=\sum_{k=1}^nF(v_k)$

 3. 计算各染色体$v_k$的选择概率$p_k$
 $$
 p_k=\frac{eval(v_k)}{Fall} \quad (k=1, 2, \cdots, n)
 $$

 4. 计算各染色体的累积概率$q_k$
 $$
 q_k=\sum_{j=1}^kp_j \quad (k=1, 2, \cdots, n)
 $$

 5. 在\[0, 1\]区间内产生一个均匀分布的伪随机数r，若$r\le q_1$ 则选择第一个染色体，否则选择第k个染色体使得$q_{k-1} < r \le q_k$成立


 #### 排序选择法

 对所有个体暗适应度大小降序排序，设计一个概率分配表将各个概率值按上述排列分配各每个个体，基于这些概率值使用轮盘赌法产生下一代群体

 #### 两两竞争法

 随机地在种群中选择k个个体进行锦标赛式的比较，选出适应值最好的个体进入下一代，复用这种方法直到下一代个体数为种群规模为止。

 #### 交叉操作

 就是让两个基因有一定的混合，因为好的基因由他好的道理，混合以后可能可能会带有两个哈基因共同的特点。
 一般来说：
 1. 可以把两个基因部分染色体交换
 2. 可以把两个基因整段染色体交换，比如中间切一刀交换

 #### 变异操作

 将个体染色体编码串中的某些基因直接随机的换成一个新的

 变异操作的方法有基本位变异、均匀变异、边界变异和非均匀变异等。
 '''
# %% [markdown]
# # 遗传算法
# ## 搜索终止条件
# 满足以下任一条件搜索就结束
#
# 1. 遗传操作中连续多次前后两代群体中最优个体的适应度相差在某一任意小的正数$\epsilon$所确定的范围内，即满足
# $$
# 0 < \left| F_{new} - F_{old} \right| < \epsilon
# $$
#
# 2. 达到最大进化代数t
#

# %% [markdown]

# ## 关键参数
# ### 种群规模
# 种群数目影响算法收敛性。数目太小不能提供足够的采样点，太大则增加计算量使得收敛时间增加。一般在20-160之间比较合适

# ### 交叉概率
# 交叉概率控制交换操作的频率，太大会使得高适应度的结构很快被破坏掉，太小则导致搜索停滞不前

# ### 变异概率
# 变异概率是增加种群多样性的第二个因素，太小不会产生新的基因块，太大则会使遗传算法编程随机搜索。
#
# %% [markdown]
# # 小结
# ## 优点
# + 可适用于灰箱甚至黑箱问题；
#
# + 搜索从群体出发，具有潜在的并行性；
#
# + 搜索使用评价函数（适应度函数）启发，过程简单；
#
# + 收敛性较强。
#
# + 具有可扩展性，容易与其他算法（粒子群、模拟退火等）结合
#
#
# ## 缺点
# + 算法参数的选择严重影响解的品质,而目前这些参数的选择大部分是依靠经验
#
# + 遗传算法的本质是随机性搜索，不能保证所得解为全局最优解
#
# + 在处理具有多个最优解的多峰问题时容易陷入局部最小值而停止搜索，造成早熟问题，无法达到全局最优
#
# %% [markdown]
# # 课堂练习
# ## 问题描述
# 假设你有900的膜当劳积分，可以兑换下面8种商品，每种商品的售价不同，需要不同积分，每种至多兑换1件。
#
# | 编号 | 积分 | 售价 |
# | ---- | ---- | ---- |
# | 0 | 99 | 9.5 |
# | 1 | 139 | 9 |
# | 2 | 300 | 19 |
# | 3 | 280 | 15 |
# | 4 | 188 | 10 |
# | 5 | 210 | 11 |
# | 6 | 240 | 12.5 |
# | 7 | 260 | 13.5 |
#
# 现在希望用900积分兑换总售价尽可能高的商品，请问该如何兑换？

# ## 问题转化
# 本问题是一个典型的01背包问题。给定$n$件物品，物品的重量为$w[i]$，物品的价值为$c[i]$。现挑选物品放入背包中，每样物品仅能取一次，假定背包能承受的最大重量为$V$，问应该如何选择装入背包中的物品，使得装入背包中物品的总价值最大？

# %%
# 染色体用整数表示，每个bit由0和1表示是否取某件物品
import numpy as np
import random
import time
from typing import List
c = 900  # 背包容量
n = 8  # 物品数量
# 物品重量
weight = [99, 139, 300, 280, 188, 210, 240, 260]
# 物品价值
value = [9.5, 9, 19, 15, 10, 11, 12.5, 13.5]

# 染色体长度，每个物品的有无对应一个独立的基因片段
chromosome_size = 8
# 染色体数目
chromo_pop_size = 100
# 选择数目
selection_num = 60
# 每代变异个体数
mutation_num = 30

last_max_value = 0
last_diff = 65536

# 判断是否退出

# 若连续两代改变量均小于5则停止迭代
# fitness list里面每个元素是[价值，和消耗积分]

class BiChromosome:
    cost_table=weight
    value_table=value
    def __init__(self,size) -> None:
        self.size=size
        self.value=0
        self.cost=0
        #random init
        self.chromosome = random.randint(0, (1 << chromosome_size) - 1)
        self.calc_fitness()
    
    def set_chromo(self,new_chromo):
        self.chromosome=new_chromo
        self.calc_fitness()

    # 计算适应度
    def calc_fitness(self):
        self.value=0
        self.cost=0
        for i in range(chromosome_size):
            mask = 1 << (chromosome_size - i - 1)
            # 看某一位是否为1，是表明取该物品，更新重量和价值的总和
            if self.chromosome & mask:
                self.cost += BiChromosome.cost_table[i]
                self.value += BiChromosome.value_table[i]
    
    # 自我变异
    def mutate(self):
        pos = random.choice(range(self.size))
        # 变异操作
        # 异或操作对某一位取反
        mask = 1 << pos
        new_chromosome = self.chromosome ^ mask
        self.set_chromo(new_chromosome)
        
    
    # 交叉
    # added_numsh是新一代的数量
    def crossover(self,mate):
        pos = random.choice(range(self.size))
        # 2号父染色体掩码，保留地位
        mask2 = (1 << (chromosome_size - pos)) - 1
        # 1号父染色体掩码，保留高位
        mask1 = (1 << chromosome_size) - 1 - mask2
        # 子代染色体，P1高位和p2低位结合
        new_chrome=(self.chromosome & mask1) + (mate.chromosome & mask2)
        next_gen=BiChromosome(self.size)
        next_gen.set_chromo(new_chrome)
        return next_gen
     
def is_finished(chromo_pop:List[BiChromosome]):
    global last_max_value
    global last_diff
    # 计算这一代的最佳适应值
    curr_max_value = 0
    for chromo in chromo_pop:
        if chromo.value > curr_max_value:
            curr_max_value = chromo.value

    # 与上一代的最佳适应度改变量
    diff = curr_max_value - last_max_value

    # 若连续两代改变量均小于5则停止迭代
    if diff < 5 and last_diff < 5:
        return True
    else:
        last_diff = diff
        last_max_value = curr_max_value
        return False

# 初始化种群
def init_population(pop_size, chromosome_size):
    chromosome_states = []
    for _ in range(pop_size):
        chromosome_states.append(BiChromosome(chromosome_size))
    return chromosome_states


# 筛选
def filter(chromo_pop:List[BiChromosome]):
    acceptable_chromosome_list = []
    # 重量超过背包容许重量的直接淘汰
    for chrome in chromo_pop:
        if chrome.cost < c:
            acceptable_chromosome_list.append(chrome)
    return acceptable_chromosome_list

# 锦标赛选择法
# 每次放回地随机挑选两个染色体，选取适应度较高的存活到下一代
def select(chromo_pop:List[BiChromosome],next_gen_num):
    survivors= []
    for _ in range(next_gen_num):
        # 随机挑选两个染色体
        candidates = random.sample(chromo_pop,2)
        # 竞争
        if candidates[0].value > candidates[1].value:
            survivors.append(candidates[0])
        else:
            survivors.append(candidates[1])
    return survivors

# 交叉
# added_num是新一代的数量
def crossover_all(chromo_pop:List[BiChromosome], added_num):
    next_gen = []
    # 父代染色体集合
    for _ in range(added_num):
        # 随机挑选两个不同的父代染色体
        parents = random.sample(chromo_pop,2)
        next_gen.append(parents[0].crossover(parents[1]))
    return next_gen

# 变异
def mutate_all(chromo_pop:List[BiChromosome],mutation_num):
    # 需要编译的数量=mutation_num
    next_gen=[]
    for _ in range(mutation_num):
        candidate = random.choice(chromo_pop)
        candidate.mutate()
    


# 迭代次数
max_iter = 200
iter = 0

# 初始化染色体
startTime = time.time()
population = init_population(chromo_pop_size, chromosome_size)

while iter < max_iter:
    if is_finished(population):
        break
    # 筛选，淘汰超出背包重量的染色体
    population = filter(population)
    # 选择
    if len(population) > selection_num:
        population = select(population,selection_num)
    # 交叉，补充新的到population数量
    population += crossover_all(population,chromo_pop_size - len(population))
    # 变异
    mutate_all(population,mutation_num)
    iter += 1
endTime = time.time()
print("time: "+str(endTime - startTime))
# 迭代结束，寻出最好的个体
population = filter(population)
best = max(population, key=lambda x: x.value)

print(bin(best.chromosome))
print('最大价值:', best.value, '所需积分:', best.cost)



# %% [markdown]
# # 习题
# ## 多重背包问题
# 物品数量：18
#
# 物品重量：8, 4, 4, 3, 5, 14, 11, 5, 4, 6, 3, 5, 12, 5, 3, 2, 9, 7
#
# 物品价值：15, 11, 5, 8, 12, 18, 20, 14, 8, 9, 6, 10, 50, 7, 2, 3, 6, 5
#
# 物品总件数：3, 4, 7, 6, 5, 3, 3, 5, 6, 4, 7, 4, 1, 7, 13, 10, 6, 9
#
# 背包承重：200
#
# 求最大价值
# %%
