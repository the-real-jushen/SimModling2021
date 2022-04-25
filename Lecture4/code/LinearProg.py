# %% [markdown]
'''
# 线性规化问题

## 什么实现新规划问题？

考虑下面这个例子：
# ![avatar](./pics/pic221.png)

线性规化问题十分常见，大多数用于资源优化分配这类问题。
'''
# %%[markdown]
# ![avatar](./pics/pic2.png)
# ## 线性规划的定义和一般形式
#
# ### 线性规划的定义
# 1. 每一个问题都用一组决策变量(x1, x2, ⋯, xn)表示某一方案, 这组决策变量的值就代表一个具体方案。一般这些变量取值是非负且**连续**的。
# 2. 存在一定的**约束条件**, 这些约束条件可以用一组**线性等式**或**线性不等式**来表示
# 3. 都有一个要求达到的**目标**, 它可用决策变量的**线性函数**(称为目标函数) 来表示。按问题的不同, 要求目标函数实现最大化或最小化。
#
#
# **满足以上三个条件**的数学模型称为**线性规划**的数学模型。
#
#
# ### 线性规划的一般形式
#
# 目标函数 $max(min) z = c_1 x_1 + c_2 x2 + \cdots + c_n x_n$
#
#
# 满足约束条件
# $$
# \begin{cases}
# a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n\leq(=, \geq)b_1\\
# a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n\leq(=, \geq)b_2\\
# \cdots \\
# a_{m1}x_1+a_{m2}x_2+\cdots+a_{mn}x_n\leq(=, \geq)b_m\\
# \end{cases}
# $$
# 即
# $$
# \begin{equation}
# max(min) cx \\
# s.t.: Ax=b \\
# Ex \leq F \\
# \underline{x} \le x \le \overline{x}
# \end{equation}
# $$

# %% [markdown]
'''
# ![avatar](./pics/pic1.png)
'''

# %% [markdown]
'''

## 用Python来解线性规划问题

下面我们用最简单的scipy来解线性规划问题，我们们就解上面的那个问题：
$$
c=-(2x_1+3X2) \\
\begin{cases}
x_1+2x_2\leq8\\
41x_1\leq16\\
41x_2\leq12\\
x_1, x_2\geq0\\
\end{cases}
$$

我们这里先用最简单的scipi来做
'''
# %%


# 目标函数
import pulp
from scipy.optimize import linprog
import numpy as np
c = [-2, -3]
# 约束条件的系数矩阵
A_ub = [[1, 2], [4, 0], [0, 4], [-1, 0], [0, -1]]
# 约束条件的上限矩阵
b_ub = [8, 16, 12, 0, 0]

res = linprog(c, A_ub, b_ub)
print(res)
# %%[markdown]
'''
## 小节一下

scipy的linprog()首先优化的是一个最小化问题，我们的问题中是一个最大化的问题，
因此我们要把目标函数乘以一个-1。

所有的约束条件，也必须写成小于等于的形式，那个_ub就是上线的意思。

这里还要注意一点的是scipy的决策变量默认是0-正无穷的，你可以修改这个上线。

上面这个没有用到等于的约束条件，我们在加几个条件试试：

$$
c=-(2x_1+3X_2) \\
\begin{cases}
x_1+2x_2\leq8\\
41x_1\leq16\\
41x_2\leq12\\
x_1+x_2=10\\
x_1, x_2\geq0\\
x_1, x_2\leq10\\
\end{cases}
$$
'''
# %%
c = [-2, -3]
# 约束条件的系数矩阵
A_ub = [[1, 2], [4, 0], [0, 4], [-1, 0], [0, -1]]
# 约束条件的上限矩阵
b_ub = [8, 16, 12, 0, 0]
# 相等条件的系数矩阵
a_eq = [[1, 1]]
# 相等条件值
b_eq = [5]
# 决策变量的上下限
bnd = [[0, 5], [0, 5]]

res = linprog(c, A_ub, b_ub, a_eq, b_eq, bnd)
print(res)
# %% [markdown]
'''
这里解释一下linprod输出的结果：
.con ：相等约束残差应该是接近0.

.fun ：目标函数的最有值（如果成功找到）.

.message :求解器返回的消息，例如出了什么问题，没有解出来.

.nit ：迭代次数.

.slack ：与约束条件测差距.

.status ：状态0-4，0.找到最优值，

.success：是否找到最优值.

.x ：解出来的决策变量的值

我们可以把上一题的条件改改，让他无解试试，大家看看他的message
'''

# %% [markdown]
'''
# quiz 1, 应用2
![avatar](./pics/pic3.png)
'''

# %% [markdown]
'''
其实scipy的linprog有很多缺陷，可以说是个玩具，他主要有以下几个问题：
1. 它不支持外部求解器（求解复杂问题时才是个问题）
2. 它不支持整数决策变量（这是最大的问题，比如上面的例子）
3. 仅仅支持写矩阵来建模，容易出错，其实也还好
4. 不支持最大化问题
5. 约束条件不支持大与等于（4，5都还好，主要是剪出来的模型描述性不高，并不影响求解）
'''

# %% [markdown]
# # 混合整数线性规划简介
# ## 分支定界法
# 分支定界法的思想主要是：把一个含整数变量规划的问题化为不含整数变量的两个子问题进行不断迭代求解。
# ![avatar](./pics/分支界定法1.png)
# ![avatar](./pics/分支界定法2.png)
'''
我们用PuLP这个包来求解一下试试，大家想用pip安装一下
'''

# %%
# 添加4个决策变量
V_NUM = 4
x = [pulp.LpVariable(f"X{i}", lowBound=0, cat=pulp.LpInteger)
     for i in range(V_NUM)]
# 初始化一个最大化的模型
model = pulp.LpProblem('应用2', pulp.LpMaximize)

# 添加限制条件
model += (1.5*x[0]+x[1]+2.4*x[2]+x[3] <= 2000, "equipA")
model += (x[0]+5*x[1]+x[2]+3.5*x[3] <= 8000, "equipB")
model += (1.5*x[0]+3*x[1]+3.5*x[2]+x[3] <= 5000, "equipC")
# 添加目标
model += 5.24*x[0]+7.3*x[1]+8.34*x[2]+4.18*x[3]
# 求解模型
status = model.solve()

# 打印出结果
for xi in model.variables():
    print(xi.name, ' = ', xi.varValue)
print("objective value: ", model.objective.value())

# %%[markdown]
'''
## 小结一下
1. PuLP解整数优化问题，只要你添加偏凉的时候指定塔的类型就行`cat=pulp.LpInteger`，
除了整数还有binary等等类型
2. 我们首先要建一个pulp的模型，这里可以指定他是最大还是最小化目标
3. 我们可以用PuLP的变量写成表达式，它的类型是`pulp.pulp.LpAffineExpression`，可以添加到
模型里面编程限制条件
4. 如果用PuLP的变量写成表达式不是相等或者比较的，她就成了目标函数
5. 对模型调用solve求解就可以得到你结果了，就在求解后的模型里面

'''
# %% [markdown]
'''
# quiz 2, 应用3
![avatar](./pics/pic4.png)
'''

# %% [markdown]
'''
# quiz 3, 应用4
![avatar](./pics/pic5.png)

决策变量或者约束条件很多的时侯，建议还是把系数写成矩阵。
那用pulp怎么做呢？其实很简单，你有了矩阵就可以用循环求和来吧所有的约束条件写出来。
'''
# %% [markdown]
'''
## 二进制规化
我们把上面这个问题在高的复杂一点：假如，方案3，4只能采用其中一个，怎么求解？

我们这个时候引入两个二进制变量就是0，1的变量y2，y3，这个代表哪种方案能采用。
两种不能同时采用那么就是y2+y3<=1.

当y2=0的时候不能采用方案3，所以可以写成x3<=y2*100，x3肯定不会大与100，
所以可以这么写。
'''

# %%
model = pulp.LpProblem('应用4', pulp.LpMinimize)
V_NUM = 5
x = [pulp.LpVariable(
    f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(V_NUM)]

y = [pulp.LpVariable(
    f'y{i}', lowBound=0, cat=pulp.LpBinary) for i in (2, 3)]
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

# 添加二进制的条件
model += (y[0]+y[1] <= 1, "not same time")
model += (x[2] <= 100*y[0], "chose x2")
model += (x[3] <= 100*y[1], "chose x3")

# 求解模型
status = model.solve()

# 打印出结果
for xi in model.variables():
    print(xi.name, ' = ', xi.varValue)
print("objective value: ", model.objective.value())


# ref https://realpython.com/linear-programming-python/

# %%
