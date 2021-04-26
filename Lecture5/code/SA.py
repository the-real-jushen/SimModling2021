# %% [markdown]
'''
 # 模拟退火算法(Simulated Annealing, SA)
 ## 基本思想
 模拟退火算法的思想源于固体的退火过程：将固体加热至足够高的温度，再缓慢冷却；升温时固体内部粒子随温度升高变为无序状，内能增加，而缓慢冷却又使得粒子逐渐趋于有序。冷却到低温时
 将达到这一低温下的内能最小状态。

其实上面说的都是忽悠你的，模拟退火也是一种解决EE dilemma的算法，就是系统一开始会比较倾向于探索，
随着系统迭代次数增多，他会慢慢的倾向于exploitation。

 ## 基本思路
 1. 设定初始温度，随机产生一个初始解$x_0$并计算相应的目标函数值$E(x_0)$。

 2. 每次迭代令$T$等于当前的温度$T_i$，对当前解$x_i$添加一个扰动，产生新解$x_j$，计算相应的目标函数值，得到$\Delta E=E(x_j)-E(x_i)$


 若$\Delta E < 0$则接收该新解，否则按概率$exp(-\Delta E / T)$接受。在当前温度下重复$L_k$次（Markov链长度）上述过程再进入下一温度直到达到设定的停止温度。
 
  模拟退火算法实际上有两层循环，在任一温度随机扰动产生新解，并计算目标函数值的变化，决定是否被接收。由于初始温度比较高，这样使E增大的新解在初始时
 也可能被接受，因而能跳出局部极小值。然后通过缓慢降低温度，算法最终可能收敛到全局最优解。
 
 ![](pics\sa.png)

 虽然低温时接收函数已经很小，但仍不排除有接收更差的解的可能，因此一般都会将退货过程中碰到的最好的可行解也记录下来，与终止算法前最后一个被接受
 的解一并输出。
 '''
# %% [markdown]
# # 模拟退火算法
# ## 参数选择
# ### 控制参数$T$的初值$T_0$
# 全局优化问题一般先进行大范围的粗略搜索再进行局部的精细搜索。只有在初始大范围搜索阶段找到全局最优解所在的区域才能逐渐缩小搜索范围最终求得全局最优解。
#
# 模拟退火算法通过控制T的初值$T_0$及其衰减变化过程来实现大范围的粗略搜索和局部的精细搜索。一般只有足够大的$T_0$才能满足算法要求。
#
# 问题规模较大时，过小的$T_0$往往导致算法难以跳出局部陷阱而达不到全局最优，但为了减少计算量，$T_0$也不宜取过大的值。
# ### $T$的衰减函数
# 常用的衰减函数有
# $$
# T_{k+1}=\alpha T_k \quad (k=0,1,2,\cdots)
# $$
# 其中$\alpha$是一个常数，可以取0.5~0.99，它的取值决定了降温的过程。
#
# 小的衰减量可能导致迭代次数增加，从而使算法进程接受更多的变换，访问更多的领域，搜索更大范围的解空间，从而返回更好的最终解。同时
# 由于在$T_k$值上已经达到准平衡，所以在$T_{k+1}$时只需少量的变化就能达到准平衡。这样就可以选择较短的Markov链来减少算法时间。
# ### Markov链长度
# Markov链长度的选取原则：在控制参数$T$的衰减函数已经确定的前提下，$L_k$应能使在控制参数$T$的每一取值上达到准平衡。从经验上来说，
# 对于简单的情况，可以令$L_k=100n$，$n$为问题规模。
# %% [markdown]
# # 模拟退火算法
# ## 收敛的一般性条件
# 收敛到全局最优的一般性条件是
#
# 1. 初始温度足够高
#
# 2. 热平衡时间足够长
#
# 3. 终止温度足够低
#
# 4. 降温过程足够缓慢
#
# 但上述条件难以在应用中同时满足
# %% [markdown]
# # 小结
# ## 优点
# + 可以突破贪心算法的局限性，以一定的概率接收较差的解，从而跳出局部最优获得全局最优解
#
# + 初始解和最终解都是随机选取的，没有关联，具有较好的鲁棒性
#
#
# ## 缺点
# + 降温速度慢会得到较好的解但会导致收敛速度下降
#
# + 降温过程过快可能得不到全局最优解
#
# %% [markdown]
# # 课堂练习
# ## 解数独
# 一个数独的解法需遵循如下规则：
#
# + 数字 1-9 在每一行只能出现一次。
# + 数字 1-9 在每一列只能出现一次。
# + 数字 1-9 在每一个以粗实线分隔的 3x3 宫格内只能出现一次。
#
# <div align=center><img src='../../pics/pic31.png' /></div>
#
# %% [markdown]
# ## 递归（解法摘自Leetcode 37.解数独）
# 我们可以考虑按照「行优先」的顺序依次枚举每一个空白格中填的数字，通过递归 + 回溯的方法枚举所有可能的填法。当递归到最后一个空白格后，如果仍然没有冲突，说明我们找到了答案；在递归的过程中，如果当前的空白格不能填下任何一个数字，那么就进行回溯。
#
# 由于每个数字在同一行、同一列、同一个九宫格中只会出现一次，因此我们可以使用$line[i], column[i], block[x][y]$分别表示第$i$行，第$j$列，第$(x, y)$个九宫格中填写数字的情况。
#
# 最容易想到的方法是用一个数组记录每个数字是否出现。由于我们可以填写的数字范围为$[1, 9]$,
# 而数组的下标从0开始，因此在存储时，我们使用一个长度为9的布尔类型的数组，其中$i$个元素的值为True，\
# 当且仅当数字$i+1$出现过。例如我们用$line[2][3]=True$表示数字4在第2行已经出现过，那么当我们在遍历到第2行的空白格时，就不能填入数字4。
#
# ### 算法
#
# 我们首先对整个数独数组进行遍历，当我们遍历到第i行第j列的位置：
#
# + 如果该位置是一个空白格，那么我们将其加入一个用来存储空白格位置的列表中，方便后续的递归操作；
# + 如果该位置是一个数字 xx，那么我们需要将$line[i][x-1], column[j][x-1], block[\lfloor i/3 \rfloor ][\lfloor j/3 \rfloor][x-1]$均置为True。
#
#
# 当我们结束了遍历过程之后，就可以开始递归枚举。当递归到第i行第j列的位置时，我们枚举填入的数字x。根据题目的要求，数字x不能和当前行、列、九宫格中已经填入的数字相同，\
# 因此$line[i][x-1], column[j][x-1], block[\lfloor i/3 \rfloor ][\lfloor j/3 \rfloor][x-1]$必须均为False。
#
# 当我们填入了数字x之后，我们要将上述的三个值都置为True，并且继续对下一个空白格位置进行递归。在回溯到当前递归层时，我们还要将上述的三个值重新置为False。
#
# %%
from simanneal import Annealer
import numpy as np
import random
import copy


class Solution:
    def solveSudoku(self, board):
        def dfs(pos: int):
            nonlocal valid
            if pos == len(spaces):
                valid = True
                return

            i, j = spaces[pos]
            for digit in range(9):
                if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                    line[i][digit] = column[j][digit] = block[i //
                                                              3][j // 3][digit] = True
                    board[i][j] = str(digit + 1)
                    dfs(pos + 1)
                    line[i][digit] = column[j][digit] = block[i //
                                                              3][j // 3][digit] = False
                if valid:
                    return

        line = [[False] * 9 for _ in range(9)]
        column = [[False] * 9 for _ in range(9)]
        block = [[[False] * 9 for _a in range(3)] for _b in range(3)]
        valid = False
        spaces = list()

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    spaces.append((i, j))
                else:
                    digit = int(board[i][j]) - 1
                    line[i][digit] = column[j][digit] = block[i //
                                                              3][j // 3][digit] = True

        dfs(0)

# %% [markdown]
# ## 模拟退火算法解数独
# 设定目标函数为所有行独立元素（每行中该元素只有一个）的个数之加上与每列独立元素个数之和。则最优目标为162$(9\times 9\times 2）$。
#
# 将整个$9\times 9$的矩阵分为9个$3\times 3$的小矩阵，每次随机挑选一个小矩阵，在小矩阵中随机挑选两个元素进行交换，得到的大矩阵作为新解用于计算下一个状态。
#
# 为什么在$3\times 3$矩阵中交换元素而不是直接在$9\times 9$矩阵中交换？因为后者不能保证每个小矩阵中各元素独立，可能导致算法不能收敛。
# %%


_ = 0
PROBLEM = np.array([
    1, _, _,  _, _, 6,  3, _, 8,
    _, _, 2,  3, _, _,  _, 9, _,
    _, _, _,  _, _, _,  7, 1, 6,

    7, _, 8,  9, 4, _,  _, _, 2,
    _, _, 4,  _, _, _,  9, _, _,
    9, _, _,  _, 2, 5,  1, _, 4,

    6, 2, 9,  _, _, _,  _, _, _,
    _, 4, _,  _, _, 7,  6, _, _,
    5, _, 7,  6, _, _,  _, _, 3,
])


def print_sudoku(state):
    border = "------+-------+------"
    rows = [state[i:i+9] for i in range(0, 81, 9)]
    for i, row in enumerate(rows):
        if i % 3 == 0:
            print(border)
        three = [row[i:i+3] for i in range(0, 9, 3)]
        print(" | ".join(
            " ".join(str(x or "_") for x in one)
            for one in three
        ))
    print(border)


def coord(row, col):
    return row * 9 + col


def block_indices(block_num):
    """return linear array indices corresp to the sq block, row major, 0-indexed.
    block:
       0 1 2     (0,0) (0,3) (0,6)
       3 4 5 --> (3,0) (3,3) (3,6)
       6 7 8     (6,0) (6,3) (6,6)
    """
    firstrow = (block_num // 3) * 3
    firstcol = (block_num % 3) * 3
    indices = [coord(firstrow + i, firstcol + j)
               for i in range(3) for j in range(3)]
    return indices


def initial_solution(problem):
    solution = problem.copy()
    for block in range(9):
        indices = block_indices(block)
        block = problem[indices]
        # 待填入元素的索引集合
        zeros = [i for i in indices if problem[i] == 0]
        # 待填入的元素
        to_fill = [i for i in range(1, 10) if i not in block]
        random.shuffle(to_fill)
        for index, value in zip(zeros, to_fill):
            solution[index] = value
    return solution


def random_move(solution, problem):
    random_solution = solution.copy()
    # 随机移动一个3x3矩阵中的两个元素
    # 选取一个3x3矩阵
    block = random.randrange(9)
    # 得到该矩阵的元素索引范围
    indices = [i for i in block_indices(block) if problem[i] == 0]
    # 随机挑选两个索引
    m, n = random.sample(indices, 2)
    # 交换
    random_solution[m], random_solution[n] = random_solution[n], random_solution[m]
    return random_solution


def calc_energy(solution):
    # 每列共有几个不同的数字
    def column_score(n): return - \
        len(set(solution[coord(i, n)] for i in range(9)))
    # 每行共有几个不同的数字
    def row_score(n): return -len(set(solution[coord(n, i)] for i in range(9)))
    # 总和
    score = sum(column_score(n) + row_score(n) for n in range(9))
    return score

# 计算接受概率


def probability(delta, T):
    return np.exp(-delta / T)


def deal(x1, x2, delta, T):
    # 求最小值，Delta < 0直接接受，> 0依概率接受
    if delta < 0:
        return x2, True
    p = probability(delta, T)
    if p > random.random():
        return x2, True
    return x1, False


def print_status(trial, accept, best):
    print('Trial:', trial, 'Accept:', accept, 'Accept Rate:', '%.2f' %
          (accept / trial), 'Best:', best)


# 初始温度
Tmax = 1
# 终止温度
Tmin = 0.1
# 温度下降率
rate = 0.99
# 马尔可夫链长度
length = 10000

T = Tmax

# 初始化解
solution = initial_solution(PROBLEM)
print_sudoku(solution)
# 最优解
best_energy = calc_energy(solution)
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
            # 已经得到最优解，提前退出
            if best_energy == -162:
                break
        random_solution = random_move(solution, PROBLEM)
        trial += 1
        random_energy = calc_energy(random_solution)
        delta = random_energy - energy
        solution, accepted = deal(solution, random_solution, delta, T)
        if accepted:
            accept += 1

    # 已经得到最优解，提前退出
    if best_energy == -162:
        break

    # 更新温度
    T *= rate
    loop_count += 1
    if loop_count % 10 == 0:
        print_status(trial, accept, best_energy)

print('-----------END----------')
print('Best Energy:', best_energy)
print_sudoku(best_solution)

# %%


class Sudoku_Sq(Annealer):
    def __init__(self, problem):
        self.problem = problem
        state = initial_solution(problem)
        super().__init__(state)

    def move(self):
        """randomly swap two cells in a random square"""
        block = random.randrange(9)
        indices = [i for i in block_indices(block) if self.problem[i] == 0]
        m, n = random.sample(indices, 2)
        self.state[m], self.state[n] = self.state[n], self.state[m]

    def energy(self):
        """calculate the number of violations: assume all rows are OK"""

        def column_score(n): return - \
            len(set(self.state[coord(i, n)] for i in range(9)))

        def row_score(n): return - \
            len(set(self.state[coord(n, i)] for i in range(9)))
        score = sum(column_score(n) + row_score(n) for n in range(9))
        if score == -162:
            self.user_exit = True  # early quit, we found a solution
        return score


sudoku = Sudoku_Sq(PROBLEM)
sudoku.copy_strategy = "method"
print_sudoku(sudoku.state)
sudoku.Tmax = 0.5
sudoku.Tmin = 0.05
sudoku.steps = 100000
state, e = sudoku.anneal()
print()
print_sudoku(state)
print("E=%f (expect -162)" % e)
# %%
