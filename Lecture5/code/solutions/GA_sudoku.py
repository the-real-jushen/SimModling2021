# %%

import numpy as np
import random


# 初始化一个数度问题
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

# 打印数独当前状态


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

# 返回某个坐标的index


def coord(row, col):
    return row * 9 + col

# 分会某一个方块的里面元素的indices


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

# 产生一个初始解


def initial_solution(problem):
    solution = problem.copy()
    for block in range(9):
        indices = block_indices(block)
        block = problem[indices]
        # 待填入元素的索引集合
        zeros = [i for i in indices if problem[i] == 0]
        # 待填入的元素，一个block里面的数字不重复
        to_fill = [i for i in range(1, 10) if i not in block]
        random.shuffle(to_fill)
        # 把每个要填的数字填到空里面
        for index, value in zip(zeros, to_fill):
            solution[index] = value
    return solution

# 对一个解进行微调
# 变异


def random_move(solution, problem):
    random_solution = solution.copy()
    # 随机移动一个3x3矩阵中的两个元素
    # 选取一个3x3矩阵
    block = random.randrange(9)
    # 得到该矩阵的元素索引范围，只选取问题中要填的元素
    indices = [i for i in block_indices(block) if problem[i] == 0]
    # 随机挑选两个索引
    m, n = random.sample(indices, 2)
    # 交换
    random_solution[m], random_solution[n] = random_solution[n], random_solution[m]
    return random_solution

# 计算出这个解的好坏，


def calc_energy(solution):
    # 每列共有几个不同的数字
    def column_score(n): return - \
        len(set(solution[coord(i, n)] for i in range(9)))
    # 每行共有几个不同的数字
    def row_score(n): return -len(set(solution[coord(n, i)] for i in range(9)))
    # 总和，每一行每一列不同数字求和，注意都去了这两个都*-1了
    score = sum(column_score(n) + row_score(n) for n in range(9))
    return score


# %%
# generic

# 交叉
# added_numsh是新一代的数量
def crossover(chromosome_states, problem, added_num):
    next_gen_chromosome_states = []
    # 父代染色体集合
    prev_gen_chromosome_states = chromosome_states.copy()
    for _ in range(added_num):
        # 随机挑选两个不同的父代染色体
        parent1Idx = random.randint(1, len(prev_gen_chromosome_states)-1)
        parent2Idx = parent1Idx
        while parent1Idx == parent2Idx:
            parent2Idx = random.randint(1, len(prev_gen_chromosome_states)-1)
        # randome exchange 1 block
        # 得到该矩阵的元素索引范围，只选取问题中要填的元素
        parent1 = prev_gen_chromosome_states[parent1Idx]
        parent2 = prev_gen_chromosome_states[parent2Idx]
        block = random.randrange(9)
        indices = [i for i in block_indices(block) if problem[i] == 0]
        next_gen = parent1
        next_gen[indices] = parent2[indices]
        next_gen_chromosome_states.append(next_gen)

    return next_gen_chromosome_states


def mutate(chromosome_states):
    # 需要编译的数量=mutation_num
    for _ in range(mutation_num):
        # 随机选择染色体
        idx = random.randint(0, len(chromosome_states)-1)
        # 变异操作
        chromosome_states[idx] = random_move(chromosome_states[idx], PROBLEM)
    return chromosome_states


def init_population(pop_size, problem):
    chromosome_states = []
    for _ in range(pop_size):
        # 随机产生染色体,这里热暗色体用的二进制编码
        chromosome = initial_solution(problem)
        chromosome_states.append(chromosome)
    return chromosome_states

# 计算适应度


def calc_fitness(chromosome_states):
    fitness_list = []
    for chromosome in chromosome_states:
        fitness = -1*calc_energy(chromosome)
        fitness_list.append(fitness)
    return fitness_list

# 筛选


# 锦标赛选择法
# 每次放回地随机挑选两个染色体，选取适应度较高的存活到下一代

def select(fitness_list, chromosome_list, selection_num):
    survivor_fitness_list = []
    survivor_chromosome_list = []
    for _ in range(selection_num):
        num = len(fitness_list)
        # 随机挑选两个染色体
        idx1 = random.randint(0, num - 1)
        idx2 = random.randint(0, num - 1)
        # 竞争
        if fitness_list[idx1] > fitness_list[idx2]:
            survivor_fitness_list.append(fitness_list[idx1])
            survivor_chromosome_list.append(chromosome_list[idx1])
        else:
            survivor_fitness_list.append(fitness_list[idx2])
            survivor_chromosome_list.append(chromosome_list[idx2])
    return survivor_fitness_list, survivor_chromosome_list


# 迭代次数
max_iter = 20
iter = 0
chromo_pop_size = 100
# 选择数目
selection_num = 60
# 每代变异个体数
mutation_num = 30

# 初始化染色体
chromosome_states = init_population(chromo_pop_size, PROBLEM)

while iter < max_iter:
    fitnesses = calc_fitness(chromosome_states)
    if np.max(fitnesses) == 162:
        break

    if len(fitnesses) > selection_num:
        fitnesses, chromosome_states = select(
            fitnesses, chromosome_states, selection_num)
    # 交叉，补充新的到population数量
    chromosome_states += crossover(chromosome_states, PROBLEM,
                                   chromo_pop_size - len(chromosome_states))
    # 变异
    chromosome_states = mutate(chromosome_states)
    iter += 1
# 迭代结束，寻出最好的个体
best_fitness = np.max(fitnesses)
idx = fitnesses.index(best_fitness)
print_sudoku(chromosome_states[idx])

# %%
