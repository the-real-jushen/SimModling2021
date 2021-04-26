# %% [markdown]
'''
## 非线性规化

对于上面的问题我们也可以坐非线性规划求解，
非线性规划问题，由于约束条件和目标函数都是五花八门，
$$ min \quad f(x) $$
$$ 
\begin{aligned}
s.t.: \quad &h(x) = 0 \\
&\underline{g} \le g(x) \le \overline{g}
\end{aligned}
$$

可以说非线性规划问题的最优解可能在其可行域中的任意一点达到。
所以没有固定的套路可解，一般来说可以用梯度下降的方法来算

![](./pics/gradient.png)

要注意的是不是所有问题都可以用梯度下来解的，
有很多目标函数没有办法求梯度，比如我们下次讲的一下优化问题。
而且即使可以求梯度，也会存在局部最优问题，要想找到全局最优也不是那么用以的。

我们这里用的scipy的optimize包，就是用的梯度下降，和之前我们界非线性方程差不多，
因此也是很容易被困在局部最优的。

我们现在尝试一下用optimize来接一下上一个需要线性化才能解的问题：
'''


# %% [markdown]
# ## 例题
# $$
# min:
# \begin{cases}
# F &= 0.0007P_{G1}^2+0.30P_{G1}+4 \\
#   &+ 0.0004P_{G2}^2+0.32P_{G2}+3
# \end{cases}
# $$
# $$
# \begin{aligned}
# st.:  & P_{G1}+P_{G2}=100 \\
# & 30 \le P_{G1} \le 150 \\
# & 0 \le P_{G2} \le 50
# \end{aligned}
# $$
#
# %%
from scipy.optimize import minimize
# 下限矩阵
lb = [30, 0]
# 上限矩阵
ub = [150, 50]
# 初值
x0 = [0, 0]
# 做成优化器能够认识的上下限
bounds = [[lb[i], ub[i]] for i in range(2)]

# 定义目标函数


def objective(x): return 0.0007 * \
    x[0] ** 2 + 0.30 * x[0] + 4 + 0.0004 * x[1] ** 2 + 0.32 * x[1] + 3


# 添加约束，这里我们就添加一个等式约束，eq表示表达式为0，ineq表示表达式大于等于0
cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] - 100}]
# 求解
res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
# 返回值和我们前面说的linprog返回的结果意思基本是一样的
print(res)

# %%
